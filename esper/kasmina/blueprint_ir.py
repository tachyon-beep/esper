"""
BlueprintIR: Universal computational graph representation.

This module implements the core intermediate representation system for
neural network computational graphs in the Kasmina Operator Subsystem.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import json


class IRNodeType(Enum):
    """Tier 1 primitive operations supported by Kasmina."""

    # Linear operations
    LINEAR = "linear"
    MATMUL = "matmul"

    # Element-wise operations
    RELU = "relu"
    GELU = "gelu"
    ADD = "add"
    MUL = "mul"
    SUB = "sub"
    DIV = "div"

    # Reduction operations
    SUM = "sum"
    MEAN = "mean"
    MAX = "max"

    # Normalization operations
    LAYER_NORM = "layer_norm"
    RMS_NORM = "rms_norm"

    # Tensor operations
    RESHAPE = "reshape"
    PERMUTE = "permute"
    CONCAT = "concat"


@dataclass
class TensorSchema:
    """Schema for tensor metadata with validation."""

    shape: Tuple[int, ...]
    dtype: str
    device: str
    requires_grad: bool = False

    SUPPORTED_DTYPES = {"float32", "float16", "bfloat16", "int32", "int64", "bool"}

    SUPPORTED_DEVICES = {"cuda", "cpu"}

    def __post_init__(self) -> None:
        """Validate tensor schema after initialization."""
        self.validate_dtype()
        self.validate_device()
        self.validate_shape()

    def validate_dtype(self) -> None:
        """Validate that dtype is supported."""
        if self.dtype not in self.SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported dtype: {self.dtype}. Supported: {self.SUPPORTED_DTYPES}"
            )

    def validate_device(self) -> None:
        """Validate that device is supported."""
        if self.device not in self.SUPPORTED_DEVICES:
            raise ValueError(
                f"Unsupported device: {self.device}. "
                f"Supported: {self.SUPPORTED_DEVICES}"
            )

    def validate_shape(self) -> None:
        """Validate that shape is valid."""
        if not all(dim > 0 for dim in self.shape):
            raise ValueError(
                f"Invalid shape: {self.shape}. All dimensions must be positive."
            )


@dataclass
class IRNode:
    """Single node in the computational graph."""

    id: str
    op_type: IRNodeType
    parameters: Dict[str, Any]
    input_nodes: List[str]
    output_shape: Optional[Tuple[int, ...]] = None
    control_flow_type: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate node after initialization."""
        if not self.id:
            raise ValueError("Node ID cannot be empty")

        if not isinstance(self.input_nodes, list):
            raise ValueError("input_nodes must be a list")


@dataclass
class BlueprintIR:
    """Complete computational graph representation."""

    nodes: Dict[str, IRNode]
    input_schemas: Dict[str, TensorSchema]
    output_schemas: Dict[str, TensorSchema]
    metadata: Dict[str, Any]

    def __post_init__(self) -> None:
        """Validate blueprint after initialization."""
        self.validate_graph_structure()

    def validate_graph_structure(self) -> None:
        """Validate that the graph structure is valid."""
        # Check that all input dependencies exist
        for node in self.nodes.values():
            for input_id in node.input_nodes:
                if input_id not in self.nodes and input_id not in self.input_schemas:
                    raise ValueError(
                        f"Node {node.id} references non-existent input {input_id}"
                    )

    def topological_sort(self) -> List[str]:
        """Return topologically sorted list of node IDs using Kahn's algorithm."""
        # Build adjacency list and in-degree count
        in_degree = dict.fromkeys(self.nodes, 0)
        adj_list: Dict[str, List[str]] = {node_id: [] for node_id in self.nodes}

        for node_id, node in self.nodes.items():
            for input_id in node.input_nodes:
                if input_id in self.nodes:  # Only internal nodes
                    adj_list[input_id].append(node_id)
                    in_degree[node_id] += 1

        # Kahn's algorithm
        queue = [node_id for node_id, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for neighbor in adj_list[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(self.nodes):
            raise ValueError("Graph contains cycles")

        return result

    def compute_canonical_hash(self) -> str:
        """Compute deterministic hash for caching."""
        # Create normalized representation
        sorted_nodes = []
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            sorted_nodes.append(
                {
                    "id": node.id,
                    "op_type": node.op_type.value,
                    "parameters": sorted(node.parameters.items())
                    if node.parameters
                    else [],
                    "input_nodes": sorted(node.input_nodes),
                    "output_shape": node.output_shape,
                    "control_flow_type": node.control_flow_type,
                }
            )

        hash_data = {
            "nodes": sorted_nodes,
            "input_schemas": {
                k: {
                    "shape": v.shape,
                    "dtype": v.dtype,
                    "device": v.device,
                    "requires_grad": v.requires_grad,
                }
                for k, v in sorted(self.input_schemas.items())
            },
            "output_schemas": {
                k: {
                    "shape": v.shape,
                    "dtype": v.dtype,
                    "device": v.device,
                    "requires_grad": v.requires_grad,
                }
                for k, v in sorted(self.output_schemas.items())
            },
            "metadata": sorted(self.metadata.items()) if self.metadata else [],
        }

        json_str = json.dumps(hash_data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(json_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage/transmission."""
        return {
            "nodes": {
                node_id: {
                    "id": node.id,
                    "op_type": node.op_type.value,
                    "parameters": node.parameters,
                    "input_nodes": node.input_nodes,
                    "output_shape": node.output_shape,
                    "control_flow_type": node.control_flow_type,
                }
                for node_id, node in self.nodes.items()
            },
            "input_schemas": {
                k: {
                    "shape": v.shape,
                    "dtype": v.dtype,
                    "device": v.device,
                    "requires_grad": v.requires_grad,
                }
                for k, v in self.input_schemas.items()
            },
            "output_schemas": {
                k: {
                    "shape": v.shape,
                    "dtype": v.dtype,
                    "device": v.device,
                    "requires_grad": v.requires_grad,
                }
                for k, v in self.output_schemas.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BlueprintIR":
        """Deserialize from dictionary."""
        nodes = {}
        for node_id, node_data in data["nodes"].items():
            nodes[node_id] = IRNode(
                id=node_data["id"],
                op_type=IRNodeType(node_data["op_type"]),
                parameters=node_data["parameters"],
                input_nodes=node_data["input_nodes"],
                output_shape=tuple(node_data["output_shape"])
                if node_data["output_shape"]
                else None,
                control_flow_type=node_data["control_flow_type"],
            )

        input_schemas = {}
        for k, v in data["input_schemas"].items():
            input_schemas[k] = TensorSchema(
                shape=tuple(v["shape"]),
                dtype=v["dtype"],
                device=v["device"],
                requires_grad=v["requires_grad"],
            )

        output_schemas = {}
        for k, v in data["output_schemas"].items():
            output_schemas[k] = TensorSchema(
                shape=tuple(v["shape"]),
                dtype=v["dtype"],
                device=v["device"],
                requires_grad=v["requires_grad"],
            )

        return cls(
            nodes=nodes,
            input_schemas=input_schemas,
            output_schemas=output_schemas,
            metadata=data["metadata"],
        )
