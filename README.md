# Esper

**Esper Morphogenetic Training Platform** - A neural network morphogenesis system enabling runtime adaptation through Just-in-Time compilation of computational graphs into optimized GPU kernels.

## Overview

Esper implements the Kasmina Operator Subsystem, which serves as the foundational execution layer for neural network adaptation. The system transforms theoretical BlueprintIR representations into high-performance executable code while maintaining strict safety guarantees and production-grade reliability.

## Key Features

- **Cache-First Architecture**: >99% cache hit ratio with <1ms lookup times
- **JIT Compilation**: Quality-first compilation with extensive optimization
- **Safety by Design**: Multiple validation layers and graceful degradation
- **GPU-Resident State**: Vectorized execution with minimal CPU overhead

## Quick Start

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (for training/inference)
- PyTorch 2.1+
- Triton 2.0+

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/esper.git
cd esper

# Create and activate virtual environment
python -m venv env
source env/bin/activate  # Linux/Mac
# or
env\Scripts\activate  # Windows

# Install dependencies
pip install -e .[dev]
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=esper --cov-report=term

# Run specific test modules
pytest tests/kasmina/test_blueprint_ir.py
```

## Development Status

This project is currently in **Phase 1: Core Infrastructure** development.

### Current Sprint: BlueprintIR Foundation (Weeks 1-2)

**Week 1 Status**: ✅ Completed

- [x] TensorSchema with dtype validation
- [x] IRNode with operation type enum  
- [x] BlueprintIR with complete metadata
- [x] IRNodeType enum with Tier 1 primitives
- [x] Unit tests for data structure validation

**Week 2 Status**: 🚧 In Progress

- [ ] Topological sorting (Kahn's algorithm) - ✅ Implemented
- [ ] Shape inference pipeline - ✅ Implemented  
- [ ] Canonical hashing for caching - ✅ Implemented
- [ ] Serialization/deserialization - ✅ Implemented
- [ ] Integration tests for graph manipulation

## Architecture

The Kasmina Operator Subsystem consists of three main planes:

1. **Compilation Plane**: BlueprintIR processing and kernel generation
2. **Caching Plane**: High-performance kernel registry and persistence
3. **Execution Plane**: GPU-resident state management and vectorized execution

## Contributing

Please see [.github/copilot-instructions.md](.github/copilot-instructions.md) for coding guidelines and best practices.

### Development Philosophy

- **Fail Hard**: Better to fail with clear errors than hide bugs
- **Explicit is Better**: No magic or implicit behaviors
- **Quality Over Speed**: Extensive optimization during compilation
- **Zero Tolerance**: No training corruption events

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
