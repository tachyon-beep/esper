#!/usr/bin/env python3
"""
Validation script for Phase 1 Week 1 deliverables.

This script validates the core BlueprintIR functionality works as expected.
"""

from esper.kasmina.blueprint_ir import TensorSchema, IRNode, BlueprintIR, IRNodeType
from esper.kasmina.shape_inference import ShapeInferenceEngine


def validate_week_1_deliverables():
    """Validate all Week 1 deliverables are working."""
    print("🚀 Validating Phase 1 Week 1 Deliverables...")

    # Test 1: TensorSchema with dtype validation
    print("\n✅ Test 1: TensorSchema validation")
    try:
        schema = TensorSchema(
            shape=(1024, 768), dtype="float32", device="cuda", requires_grad=True
        )
        print(f"   Created schema: {schema.shape} {schema.dtype} on {schema.device}")

        # Test invalid dtype (should raise)
        try:
            TensorSchema(shape=(10,), dtype="invalid", device="cuda")
            print("   ❌ ERROR: Invalid dtype should have raised exception")
            return False
        except ValueError:
            print("   ✓ Invalid dtype properly rejected")

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

    # Test 2: IRNode with operation type enum
    print("\n✅ Test 2: IRNode creation")
    try:
        node = IRNode(
            id="linear_layer_1",
            op_type=IRNodeType.LINEAR,
            parameters={"out_features": 768, "bias": True},
            input_nodes=["input_tensor"],
            output_shape=(1024, 768),
        )
        print(f"   Created node: {node.id} ({node.op_type.value})")
        print(f"   Parameters: {node.parameters}")
        print(f"   Input dependencies: {node.input_nodes}")

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

    # Test 3: BlueprintIR with complete metadata
    print("\n✅ Test 3: BlueprintIR creation and validation")
    try:
        # Create a simple 2-layer network: Input -> Linear -> ReLU -> Output
        input_schema = TensorSchema(shape=(32, 512), dtype="float32", device="cuda")

        linear_node = IRNode(
            id="linear_1",
            op_type=IRNodeType.LINEAR,
            parameters={"out_features": 768},
            input_nodes=["input_data"],
        )

        relu_node = IRNode(
            id="relu_1",
            op_type=IRNodeType.RELU,
            parameters={},
            input_nodes=["linear_1"],
        )

        output_schema = TensorSchema(shape=(32, 768), dtype="float32", device="cuda")

        blueprint = BlueprintIR(
            nodes={"linear_1": linear_node, "relu_1": relu_node},
            input_schemas={"input_data": input_schema},
            output_schemas={"output_data": output_schema},
            metadata={
                "version": "1.0",
                "created_by": "validation_script",
                "description": "Simple 2-layer network for testing",
            },
        )

        print(f"   Created blueprint with {len(blueprint.nodes)} nodes")
        print(f"   Metadata: {blueprint.metadata}")

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

    # Test 4: IRNodeType enum with Tier 1 primitives
    print("\n✅ Test 4: IRNodeType enum completeness")
    expected_ops = [
        "LINEAR",
        "MATMUL",
        "RELU",
        "GELU",
        "ADD",
        "MUL",
        "SUB",
        "DIV",
        "SUM",
        "MEAN",
        "MAX",
        "LAYER_NORM",
        "RMS_NORM",
        "RESHAPE",
        "PERMUTE",
        "CONCAT",
    ]

    available_ops = [op.name for op in IRNodeType]
    print(f"   Available operations: {len(available_ops)}")

    missing_ops = set(expected_ops) - set(available_ops)
    if missing_ops:
        print(f"   ❌ Missing operations: {missing_ops}")
        return False
    else:
        print(f"   ✓ All {len(expected_ops)} Tier 1 primitives available")

    # Test 5: Topological sorting
    print("\n✅ Test 5: Graph operations")
    try:
        sorted_nodes = blueprint.topological_sort()
        print(f"   Topological order: {sorted_nodes}")

        # Should be linear_1 -> relu_1
        if sorted_nodes != ["linear_1", "relu_1"]:
            print("   ❌ Wrong topological order: expected ['linear_1', 'relu_1']")
            return False
        else:
            print("   ✓ Correct topological ordering")

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

    # Test 6: Canonical hashing
    print("\n✅ Test 6: Canonical hashing for caching")
    try:
        hash1 = blueprint.compute_canonical_hash()
        print(f"   Generated hash: {hash1[:16]}...")

        # Create identical blueprint
        blueprint2 = BlueprintIR.from_dict(blueprint.to_dict())
        hash2 = blueprint2.compute_canonical_hash()

        if hash1 != hash2:
            print("   ❌ Identical blueprints have different hashes")
            return False
        else:
            print("   ✓ Deterministic hashing working")

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

    # Test 7: Serialization/deserialization
    print("\n✅ Test 7: Serialization roundtrip")
    try:
        data = blueprint.to_dict()
        restored = BlueprintIR.from_dict(data)

        if len(restored.nodes) != len(blueprint.nodes):
            print("   ❌ Node count mismatch after deserialization")
            return False

        if restored.compute_canonical_hash() != blueprint.compute_canonical_hash():
            print("   ❌ Hash mismatch after deserialization")
            return False

        print("   ✓ Serialization roundtrip successful")

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

    # Test 8: Shape inference integration
    print("\n✅ Test 8: Shape inference integration")
    try:
        engine = ShapeInferenceEngine()

        # Test linear layer shape inference
        linear_output = engine.infer_shape(
            IRNodeType.LINEAR, [(32, 512)], {"out_features": 768}
        )

        if linear_output != (32, 768):
            print(f"   ❌ Wrong linear output shape: {linear_output}")
            return False

        # Test ReLU (elementwise) shape inference
        relu_output = engine.infer_shape(IRNodeType.RELU, [(32, 768)], {})

        if relu_output != (32, 768):
            print(f"   ❌ Wrong ReLU output shape: {relu_output}")
            return False

        print("   ✓ Shape inference working for basic operations")

    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False

    print("\n🎉 All Phase 1 Week 1 deliverables validated successfully!")
    print("\n📋 Summary:")
    print("   ✅ TensorSchema with dtype validation")
    print("   ✅ IRNode with operation type enum")
    print("   ✅ BlueprintIR with complete metadata")
    print("   ✅ IRNodeType enum with Tier 1 primitives")
    print("   ✅ Graph structure validation")
    print("   ✅ Topological sorting (Kahn's algorithm)")
    print("   ✅ Shape inference pipeline")
    print("   ✅ Canonical hashing for caching")
    print("   ✅ Serialization/deserialization")

    return True


if __name__ == "__main__":
    success = validate_week_1_deliverables()
    exit(0 if success else 1)
