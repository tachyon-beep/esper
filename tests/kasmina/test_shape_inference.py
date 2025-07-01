"""
Tests for shape inference engine.

Test coverage for all supported operations and edge cases.
"""

import pytest
from esper.kasmina.shape_inference import ShapeInferenceEngine, ShapeInferenceError
from esper.kasmina.blueprint_ir import IRNodeType


class TestShapeInferenceEngine:
    """Test shape inference for all supported operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ShapeInferenceEngine()

    def test_linear_shape_inference(self):
        """Test linear layer shape inference."""
        input_shapes = [(10, 512)]
        parameters = {"out_features": 768}

        output_shape = self.engine.infer_shape(
            IRNodeType.LINEAR, input_shapes, parameters
        )

        assert output_shape == (10, 768)

    def test_linear_batch_dimensions(self):
        """Test linear layer with batch dimensions."""
        input_shapes = [(32, 128, 512)]
        parameters = {"out_features": 768}

        output_shape = self.engine.infer_shape(
            IRNodeType.LINEAR, input_shapes, parameters
        )

        assert output_shape == (32, 128, 768)

    def test_linear_missing_parameter_raises_error(self):
        """Test that missing out_features raises error."""
        input_shapes = [(10, 512)]
        parameters = {}

        with pytest.raises(ShapeInferenceError, match="missing 'out_features'"):
            self.engine.infer_shape(IRNodeType.LINEAR, input_shapes, parameters)

    def test_matmul_shape_inference(self):
        """Test matrix multiplication shape inference."""
        input_shapes = [(10, 512), (512, 768)]
        parameters = {}

        output_shape = self.engine.infer_shape(
            IRNodeType.MATMUL, input_shapes, parameters
        )

        assert output_shape == (10, 768)

    def test_matmul_batch_dimensions(self):
        """Test matmul with batch dimensions."""
        input_shapes = [(32, 10, 512), (32, 512, 768)]
        parameters = {}

        output_shape = self.engine.infer_shape(
            IRNodeType.MATMUL, input_shapes, parameters
        )

        assert output_shape == (32, 10, 768)

    def test_matmul_dimension_mismatch_raises_error(self):
        """Test that dimension mismatch raises error."""
        input_shapes = [(10, 512), (256, 768)]  # 512 != 256
        parameters = {}

        with pytest.raises(ShapeInferenceError, match="dimension mismatch"):
            self.engine.infer_shape(IRNodeType.MATMUL, input_shapes, parameters)

    @pytest.mark.parametrize(
        "op_type",
        [IRNodeType.RELU, IRNodeType.GELU, IRNodeType.LAYER_NORM, IRNodeType.RMS_NORM],
    )
    def test_elementwise_unary_operations(self, op_type):
        """Test unary element-wise operations."""
        input_shapes = [(32, 128, 768)]
        parameters = {}

        output_shape = self.engine.infer_shape(op_type, input_shapes, parameters)

        assert output_shape == (32, 128, 768)

    @pytest.mark.parametrize(
        "op_type", [IRNodeType.ADD, IRNodeType.MUL, IRNodeType.SUB, IRNodeType.DIV]
    )
    def test_elementwise_binary_operations(self, op_type):
        """Test binary element-wise operations."""
        input_shapes = [(32, 128, 768), (32, 128, 768)]
        parameters = {}

        output_shape = self.engine.infer_shape(op_type, input_shapes, parameters)

        assert output_shape == (32, 128, 768)

    def test_elementwise_broadcasting(self):
        """Test element-wise operations with broadcasting."""
        input_shapes = [(32, 128, 768), (1, 128, 1)]
        parameters = {}

        output_shape = self.engine.infer_shape(IRNodeType.ADD, input_shapes, parameters)

        assert output_shape == (32, 128, 768)

    @pytest.mark.parametrize(
        "op_type", [IRNodeType.SUM, IRNodeType.MEAN, IRNodeType.MAX]
    )
    def test_reduction_all_dimensions(self, op_type):
        """Test reduction over all dimensions."""
        input_shapes = [(32, 128, 768)]
        parameters = {}

        output_shape = self.engine.infer_shape(op_type, input_shapes, parameters)

        assert output_shape == ()

    @pytest.mark.parametrize(
        "op_type", [IRNodeType.SUM, IRNodeType.MEAN, IRNodeType.MAX]
    )
    def test_reduction_single_dimension(self, op_type):
        """Test reduction over single dimension."""
        input_shapes = [(32, 128, 768)]
        parameters = {"dim": 1}

        output_shape = self.engine.infer_shape(op_type, input_shapes, parameters)

        assert output_shape == (32, 768)

    @pytest.mark.parametrize(
        "op_type", [IRNodeType.SUM, IRNodeType.MEAN, IRNodeType.MAX]
    )
    def test_reduction_keepdim(self, op_type):
        """Test reduction with keepdim=True."""
        input_shapes = [(32, 128, 768)]
        parameters = {"dim": 1, "keepdim": True}

        output_shape = self.engine.infer_shape(op_type, input_shapes, parameters)

        assert output_shape == (32, 1, 768)

    def test_reshape_operation(self):
        """Test reshape operation."""
        input_shapes = [(32, 768)]
        parameters = {"shape": (8, 4, 768)}

        output_shape = self.engine.infer_shape(
            IRNodeType.RESHAPE, input_shapes, parameters
        )

        assert output_shape == (8, 4, 768)

    def test_reshape_with_negative_one(self):
        """Test reshape with -1 dimension."""
        input_shapes = [(32, 768)]
        parameters = {"shape": (-1, 768)}

        output_shape = self.engine.infer_shape(
            IRNodeType.RESHAPE, input_shapes, parameters
        )

        assert output_shape == (32, 768)

    def test_reshape_size_mismatch_raises_error(self):
        """Test that reshape size mismatch raises error."""
        input_shapes = [(32, 768)]
        parameters = {"shape": (16, 768)}  # 32*768 != 16*768

        with pytest.raises(ShapeInferenceError, match="size mismatch"):
            self.engine.infer_shape(IRNodeType.RESHAPE, input_shapes, parameters)

    def test_permute_operation(self):
        """Test permute operation."""
        input_shapes = [(32, 128, 768)]
        parameters = {"dims": [2, 0, 1]}  # (768, 32, 128)

        output_shape = self.engine.infer_shape(
            IRNodeType.PERMUTE, input_shapes, parameters
        )

        assert output_shape == (768, 32, 128)

    def test_permute_invalid_dims_raises_error(self):
        """Test that invalid permutation raises error."""
        input_shapes = [(32, 128, 768)]
        parameters = {"dims": [0, 1, 3]}  # Invalid: no dimension 3

        with pytest.raises(ShapeInferenceError, match="Invalid permutation"):
            self.engine.infer_shape(IRNodeType.PERMUTE, input_shapes, parameters)

    def test_concat_operation(self):
        """Test concatenation operation."""
        input_shapes = [(32, 128, 768), (32, 256, 768)]
        parameters = {"dim": 1}

        output_shape = self.engine.infer_shape(
            IRNodeType.CONCAT, input_shapes, parameters
        )

        assert output_shape == (32, 384, 768)

    def test_concat_dimension_mismatch_raises_error(self):
        """Test that concat dimension mismatch raises error."""
        input_shapes = [(32, 128, 768), (16, 128, 768)]  # Different batch size
        parameters = {"dim": 1}

        with pytest.raises(ShapeInferenceError, match="first input size"):
            self.engine.infer_shape(IRNodeType.CONCAT, input_shapes, parameters)

    def test_unsupported_operation_raises_error(self):
        """Test that unsupported operation raises error."""
        # Create a new enum value that's not in the inference functions
        input_shapes = [(10, 512)]
        parameters = {}

        # This should raise an error since we don't have an inference function
        with pytest.raises(ShapeInferenceError, match="No shape inference"):
            # Use a made-up operation type by directly creating enum
            fake_op = IRNodeType.LINEAR  # Use existing but call with wrong name
            # Temporarily remove it from the functions dict
            original_func = self.engine._inference_functions.pop(IRNodeType.LINEAR)
            try:
                self.engine.infer_shape(fake_op, input_shapes, parameters)
            finally:
                # Restore the function
                self.engine._inference_functions[IRNodeType.LINEAR] = original_func


class TestBroadcastShapes:
    """Test shape broadcasting functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ShapeInferenceEngine()

    def test_broadcast_identical_shapes(self):
        """Test broadcasting identical shapes."""
        result = self.engine._broadcast_shapes((32, 128), (32, 128))
        assert result == (32, 128)

    def test_broadcast_with_ones(self):
        """Test broadcasting with dimension 1."""
        result = self.engine._broadcast_shapes((32, 1), (1, 128))
        assert result == (32, 128)

    def test_broadcast_different_ranks(self):
        """Test broadcasting different rank tensors."""
        result = self.engine._broadcast_shapes((128,), (32, 128))
        assert result == (32, 128)

    def test_broadcast_incompatible_raises_error(self):
        """Test that incompatible shapes raise error."""
        with pytest.raises(ValueError, match="Cannot broadcast"):
            self.engine._broadcast_shapes((32, 64), (32, 128))
