"""
Shape inference system for BlueprintIR operations.

Provides shape computation for all supported operation types.
"""

from typing import Dict, List, Tuple, Any, Callable
from .blueprint_ir import IRNodeType


class ShapeInferenceError(Exception):
    """Raised when shape inference fails."""

    pass


class ShapeInferenceEngine:
    """Engine for computing output shapes from operation inputs."""

    def __init__(self):
        self._inference_functions: Dict[IRNodeType, Callable] = {
            IRNodeType.LINEAR: self._infer_linear_shape,
            IRNodeType.MATMUL: self._infer_matmul_shape,
            IRNodeType.RELU: self._infer_elementwise_shape,
            IRNodeType.GELU: self._infer_elementwise_shape,
            IRNodeType.ADD: self._infer_elementwise_shape,
            IRNodeType.MUL: self._infer_elementwise_shape,
            IRNodeType.SUB: self._infer_elementwise_shape,
            IRNodeType.DIV: self._infer_elementwise_shape,
            IRNodeType.SUM: self._infer_reduction_shape,
            IRNodeType.MEAN: self._infer_reduction_shape,
            IRNodeType.MAX: self._infer_reduction_shape,
            IRNodeType.LAYER_NORM: self._infer_elementwise_shape,
            IRNodeType.RMS_NORM: self._infer_elementwise_shape,
            IRNodeType.RESHAPE: self._infer_reshape_shape,
            IRNodeType.PERMUTE: self._infer_permute_shape,
            IRNodeType.CONCAT: self._infer_concat_shape,
        }

    def infer_shape(
        self,
        op_type: IRNodeType,
        input_shapes: List[Tuple[int, ...]],
        parameters: Dict[str, Any],
    ) -> Tuple[int, ...]:
        """Infer output shape for given operation."""
        if op_type not in self._inference_functions:
            raise ShapeInferenceError(f"No shape inference for operation: {op_type}")

        try:
            return self._inference_functions[op_type](input_shapes, parameters)
        except Exception as e:
            raise ShapeInferenceError(
                f"Shape inference failed for {op_type}: {e}"
            ) from e

    def _infer_linear_shape(
        self, input_shapes: List[Tuple[int, ...]], parameters: Dict[str, Any]
    ) -> Tuple[int, ...]:
        """Infer shape for linear/dense layer."""
        if len(input_shapes) != 1:
            raise ShapeInferenceError("Linear layer requires exactly one input")

        input_shape = input_shapes[0]
        if len(input_shape) < 2:
            raise ShapeInferenceError(
                "Linear layer input must have at least 2 dimensions"
            )

        out_features = parameters.get("out_features")
        if out_features is None:
            raise ShapeInferenceError("Linear layer missing 'out_features' parameter")

        return input_shape[:-1] + (out_features,)

    def _infer_matmul_shape(
        self, input_shapes: List[Tuple[int, ...]], parameters: Dict[str, Any]
    ) -> Tuple[int, ...]:
        """Infer shape for matrix multiplication."""
        if len(input_shapes) != 2:
            raise ShapeInferenceError(
                "Matrix multiplication requires exactly two inputs"
            )

        shape_a, shape_b = input_shapes

        if len(shape_a) < 2 or len(shape_b) < 2:
            raise ShapeInferenceError(
                "Matrix multiplication inputs must have at least 2 dimensions"
            )

        # Check that inner dimensions match
        if shape_a[-1] != shape_b[-2]:
            raise ShapeInferenceError(
                f"Matrix multiplication dimension mismatch: {shape_a[-1]} != {shape_b[-2]}"
            )

        # Broadcast batch dimensions
        batch_dims_a = shape_a[:-2]
        batch_dims_b = shape_b[:-2]

        try:
            batch_dims = self._broadcast_shapes(batch_dims_a, batch_dims_b)
        except ValueError as e:
            raise ShapeInferenceError(f"Batch dimension broadcast failed: {e}") from e

        return batch_dims + (shape_a[-2], shape_b[-1])

    def _infer_elementwise_shape(
        self, input_shapes: List[Tuple[int, ...]], parameters: Dict[str, Any]
    ) -> Tuple[int, ...]:
        """Infer shape for element-wise operations."""
        if len(input_shapes) == 1:
            return input_shapes[0]

        if len(input_shapes) == 2:
            try:
                return self._broadcast_shapes(input_shapes[0], input_shapes[1])
            except ValueError as e:
                raise ShapeInferenceError(f"Element-wise broadcast failed: {e}") from e

        raise ShapeInferenceError("Element-wise operations support 1 or 2 inputs")

    def _infer_reduction_shape(
        self, input_shapes: List[Tuple[int, ...]], parameters: Dict[str, Any]
    ) -> Tuple[int, ...]:
        """Infer shape for reduction operations."""
        if len(input_shapes) != 1:
            raise ShapeInferenceError("Reduction operations require exactly one input")

        input_shape = input_shapes[0]
        dim = parameters.get("dim")
        keepdim = parameters.get("keepdim", False)

        if dim is None:
            # Reduce all dimensions
            return (1,) if keepdim else ()

        if isinstance(dim, int):
            dim = [dim]

        # Normalize negative dimensions
        normalized_dims = []
        for d in dim:
            if d < 0:
                d = len(input_shape) + d
            if d < 0 or d >= len(input_shape):
                raise ShapeInferenceError(
                    f"Dimension {d} out of range for shape {input_shape}"
                )
            normalized_dims.append(d)

        if keepdim:
            result_shape = list(input_shape)
            for d in normalized_dims:
                result_shape[d] = 1
            return tuple(result_shape)
        else:
            result_shape = []
            for i, size in enumerate(input_shape):
                if i not in normalized_dims:
                    result_shape.append(size)
            return tuple(result_shape) if result_shape else ()

    def _infer_reshape_shape(
        self, input_shapes: List[Tuple[int, ...]], parameters: Dict[str, Any]
    ) -> Tuple[int, ...]:
        """Infer shape for reshape operation."""
        if len(input_shapes) != 1:
            raise ShapeInferenceError("Reshape requires exactly one input")

        target_shape = parameters.get("shape")
        if target_shape is None:
            raise ShapeInferenceError("Reshape missing 'shape' parameter")

        input_shape = input_shapes[0]
        input_numel = 1
        for dim in input_shape:
            input_numel *= dim

        # Handle -1 in target shape
        target_shape = list(target_shape)
        unknown_dim_idx = None
        target_numel = 1

        for i, dim in enumerate(target_shape):
            if dim == -1:
                if unknown_dim_idx is not None:
                    raise ShapeInferenceError("Only one dimension can be -1 in reshape")
                unknown_dim_idx = i
            else:
                target_numel *= dim

        if unknown_dim_idx is not None:
            if target_numel == 0:
                raise ShapeInferenceError(
                    "Cannot infer dimension when other dimensions are 0"
                )
            target_shape[unknown_dim_idx] = input_numel // target_numel
            target_numel = input_numel

        if input_numel != target_numel:
            raise ShapeInferenceError(
                f"Reshape size mismatch: {input_numel} != {target_numel}"
            )

        return tuple(target_shape)

    def _infer_permute_shape(
        self, input_shapes: List[Tuple[int, ...]], parameters: Dict[str, Any]
    ) -> Tuple[int, ...]:
        """Infer shape for permute operation."""
        if len(input_shapes) != 1:
            raise ShapeInferenceError("Permute requires exactly one input")

        dims = parameters.get("dims")
        if dims is None:
            raise ShapeInferenceError("Permute missing 'dims' parameter")

        input_shape = input_shapes[0]

        if len(dims) != len(input_shape):
            raise ShapeInferenceError(
                f"Permute dims length {len(dims)} != input rank {len(input_shape)}"
            )

        # Check that dims is a valid permutation
        if sorted(dims) != list(range(len(input_shape))):
            raise ShapeInferenceError(f"Invalid permutation: {dims}")

        return tuple(input_shape[i] for i in dims)

    def _infer_concat_shape(
        self, input_shapes: List[Tuple[int, ...]], parameters: Dict[str, Any]
    ) -> Tuple[int, ...]:
        """Infer shape for concatenation operation."""
        if len(input_shapes) < 2:
            raise ShapeInferenceError("Concat requires at least two inputs")

        dim = parameters.get("dim", 0)
        first_shape = input_shapes[0]

        # Normalize negative dimension
        if dim < 0:
            dim = len(first_shape) + dim

        if dim < 0 or dim >= len(first_shape):
            raise ShapeInferenceError(f"Concat dimension {dim} out of range")

        # Check that all shapes match except in concat dimension
        result_shape = list(first_shape)
        concat_size = first_shape[dim]

        for i, shape in enumerate(input_shapes[1:], 1):
            if len(shape) != len(first_shape):
                raise ShapeInferenceError(
                    f"Input {i} rank {len(shape)} != first input rank {len(first_shape)}"
                )

            for j, (s1, s2) in enumerate(zip(first_shape, shape)):
                if j == dim:
                    concat_size += s2
                elif s1 != s2:
                    raise ShapeInferenceError(
                        f"Input {i} dimension {j} size {s2} != first input size {s1}"
                    )

        result_shape[dim] = concat_size
        return tuple(result_shape)

    def _broadcast_shapes(
        self, shape1: Tuple[int, ...], shape2: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        """Broadcast two shapes together."""
        # Pad shorter shape with 1s on the left
        len1, len2 = len(shape1), len(shape2)
        max_len = max(len1, len2)

        padded1 = (1,) * (max_len - len1) + shape1
        padded2 = (1,) * (max_len - len2) + shape2

        result = []
        for s1, s2 in zip(padded1, padded2):
            if s1 == 1:
                result.append(s2)
            elif s2 == 1:
                result.append(s1)
            elif s1 == s2:
                result.append(s1)
            else:
                raise ValueError(f"Cannot broadcast dimensions {s1} and {s2}")

        return tuple(result)
