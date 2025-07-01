"""
Tests for BlueprintIR core data structures.

Test coverage for TensorSchema, IRNode, and BlueprintIR validation.
"""

import pytest
from esper.kasmina.blueprint_ir import TensorSchema, IRNode, BlueprintIR, IRNodeType


class TestTensorSchema:
    """Test TensorSchema validation and functionality."""

    def test_valid_tensor_schema(self):
        """Test creation of valid tensor schema."""
        schema = TensorSchema(
            shape=(1024, 768), dtype="float32", device="cuda", requires_grad=True
        )
        assert schema.shape == (1024, 768)
        assert schema.dtype == "float32"
        assert schema.device == "cuda"
        assert schema.requires_grad is True

    def test_invalid_dtype_raises_error(self):
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported dtype"):
            TensorSchema(shape=(1024, 768), dtype="invalid_dtype", device="cuda")

    def test_invalid_device_raises_error(self):
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported device"):
            TensorSchema(shape=(1024, 768), dtype="float32", device="invalid_device")

    def test_invalid_shape_raises_error(self):
        """Test that invalid shape raises ValueError."""
        with pytest.raises(ValueError, match="Invalid shape"):
            TensorSchema(shape=(1024, 0, 768), dtype="float32", device="cuda")

    @pytest.mark.parametrize(
        "dtype", ["float32", "float16", "bfloat16", "int32", "int64", "bool"]
    )
    def test_all_supported_dtypes(self, dtype):
        """Test that all supported dtypes work."""
        schema = TensorSchema(shape=(1024,), dtype=dtype, device="cuda")
        assert schema.dtype == dtype

    @pytest.mark.parametrize("device", ["cuda", "cpu"])
    def test_all_supported_devices(self, device):
        """Test that all supported devices work."""
        schema = TensorSchema(shape=(1024,), dtype="float32", device=device)
        assert schema.device == device


class TestIRNode:
    """Test IRNode validation and functionality."""

    def test_valid_ir_node(self):
        """Test creation of valid IR node."""
        node = IRNode(
            id="node_1",
            op_type=IRNodeType.LINEAR,
            parameters={"out_features": 768},
            input_nodes=["input_1"],
            output_shape=(1024, 768),
        )
        assert node.id == "node_1"
        assert node.op_type == IRNodeType.LINEAR
        assert node.parameters == {"out_features": 768}
        assert node.input_nodes == ["input_1"]
        assert node.output_shape == (1024, 768)

    def test_empty_id_raises_error(self):
        """Test that empty ID raises ValueError."""
        with pytest.raises(ValueError, match="Node ID cannot be empty"):
            IRNode(id="", op_type=IRNodeType.LINEAR, parameters={}, input_nodes=[])

    def test_invalid_input_nodes_raises_error(self):
        """Test that non-list input_nodes raises ValueError."""
        with pytest.raises(ValueError, match="input_nodes must be a list"):
            IRNode(
                id="node_1",
                op_type=IRNodeType.LINEAR,
                parameters={},
                input_nodes="not_a_list",  # type: ignore
            )


class TestBlueprintIR:
    """Test BlueprintIR validation and functionality."""

    def test_valid_blueprint_ir(self):
        """Test creation of valid BlueprintIR."""
        input_schema = TensorSchema(shape=(1024, 512), dtype="float32", device="cuda")
        output_schema = TensorSchema(shape=(1024, 768), dtype="float32", device="cuda")

        node = IRNode(
            id="linear_1",
            op_type=IRNodeType.LINEAR,
            parameters={"out_features": 768},
            input_nodes=["input_1"],
            output_shape=(1024, 768),
        )

        blueprint = BlueprintIR(
            nodes={"linear_1": node},
            input_schemas={"input_1": input_schema},
            output_schemas={"output_1": output_schema},
            metadata={"version": "1.0"},
        )

        assert "linear_1" in blueprint.nodes
        assert "input_1" in blueprint.input_schemas
        assert "output_1" in blueprint.output_schemas
        assert blueprint.metadata["version"] == "1.0"

    def test_invalid_dependency_raises_error(self):
        """Test that invalid node dependency raises ValueError."""
        node = IRNode(
            id="linear_1",
            op_type=IRNodeType.LINEAR,
            parameters={"out_features": 768},
            input_nodes=["nonexistent_input"],
        )

        with pytest.raises(ValueError, match="references non-existent input"):
            BlueprintIR(
                nodes={"linear_1": node},
                input_schemas={},
                output_schemas={},
                metadata={},
            )

    def test_topological_sort_simple_chain(self):
        """Test topological sort on simple chain."""
        node1 = IRNode(
            id="node_1",
            op_type=IRNodeType.LINEAR,
            parameters={},
            input_nodes=["input_1"],
        )
        node2 = IRNode(
            id="node_2", op_type=IRNodeType.RELU, parameters={}, input_nodes=["node_1"]
        )

        blueprint = BlueprintIR(
            nodes={"node_1": node1, "node_2": node2},
            input_schemas={"input_1": TensorSchema((10,), "float32", "cuda")},
            output_schemas={},
            metadata={},
        )

        sorted_nodes = blueprint.topological_sort()
        assert sorted_nodes == ["node_1", "node_2"]

    def test_topological_sort_cycle_raises_error(self):
        """Test that cyclic graph raises ValueError."""
        node1 = IRNode(
            id="node_1",
            op_type=IRNodeType.ADD,
            parameters={},
            input_nodes=["node_2", "input_1"],
        )
        node2 = IRNode(
            id="node_2", op_type=IRNodeType.MUL, parameters={}, input_nodes=["node_1"]
        )

        blueprint = BlueprintIR(
            nodes={"node_1": node1, "node_2": node2},
            input_schemas={"input_1": TensorSchema((10,), "float32", "cuda")},
            output_schemas={},
            metadata={},
        )

        with pytest.raises(ValueError, match="Graph contains cycles"):
            blueprint.topological_sort()

    def test_canonical_hash_deterministic(self):
        """Test that canonical hash is deterministic."""

        def create_blueprint():
            node = IRNode(
                id="node_1",
                op_type=IRNodeType.LINEAR,
                parameters={"out_features": 768},
                input_nodes=["input_1"],
            )
            return BlueprintIR(
                nodes={"node_1": node},
                input_schemas={"input_1": TensorSchema((10, 512), "float32", "cuda")},
                output_schemas={"output_1": TensorSchema((10, 768), "float32", "cuda")},
                metadata={"version": "1.0"},
            )

        blueprint1 = create_blueprint()
        blueprint2 = create_blueprint()

        hash1 = blueprint1.compute_canonical_hash()
        hash2 = blueprint2.compute_canonical_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex length

    def test_serialization_roundtrip(self):
        """Test that serialization and deserialization work correctly."""
        original = BlueprintIR(
            nodes={
                "node_1": IRNode(
                    id="node_1",
                    op_type=IRNodeType.LINEAR,
                    parameters={"out_features": 768},
                    input_nodes=["input_1"],
                    output_shape=(10, 768),
                )
            },
            input_schemas={"input_1": TensorSchema((10, 512), "float32", "cuda", True)},
            output_schemas={
                "output_1": TensorSchema((10, 768), "float32", "cuda", False)
            },
            metadata={"version": "1.0", "author": "test"},
        )

        # Serialize to dict
        data = original.to_dict()

        # Deserialize from dict
        restored = BlueprintIR.from_dict(data)

        # Check that everything matches
        assert len(restored.nodes) == len(original.nodes)
        assert "node_1" in restored.nodes

        restored_node = restored.nodes["node_1"]
        original_node = original.nodes["node_1"]

        assert restored_node.id == original_node.id
        assert restored_node.op_type == original_node.op_type
        assert restored_node.parameters == original_node.parameters
        assert restored_node.input_nodes == original_node.input_nodes
        assert restored_node.output_shape == original_node.output_shape

        assert restored.input_schemas.keys() == original.input_schemas.keys()
        assert restored.output_schemas.keys() == original.output_schemas.keys()
        assert restored.metadata == original.metadata

        # Hashes should be identical
        assert restored.compute_canonical_hash() == original.compute_canonical_hash()
