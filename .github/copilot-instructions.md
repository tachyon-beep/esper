# Copilot Coding Instructions for the Esper Project

This document provides guidelines for contributing to the `esper` project. As an AI assistant, please adhere to these principles to ensure the codebase is robust, maintainable, and aligns with the project's architectural philosophy.

## 1. Python Development Environment

- **Python Version**: The project uses Python 3.12 or newer. Please leverage features available in this version, such as structural pattern matching, the `|` operator for union types, and improved type hinting.
- **Style Guide**: All Python code must adhere to [PEP 8](https://www.python.org/dev/peps/pep-0008/).
- **Type Hinting**: All new functions and methods must include type hints for arguments and return values. This is critical for static analysis and maintaining a clear API.
- **Readability**: Write code that is clear and self-documenting.

## 2. Testing with Pytest

- **Framework**: We use `pytest` for testing. All tests should be written using `pytest`'s features.
- **Test Location**: Tests are located in the `tests/` directory and should mirror the structure of the `esper/` package.
- **Assertions**: Use plain `assert` statements for assertions.
- **Fixtures**: Use `pytest` fixtures for setting up and tearing down test state. This is preferred over `unittest`-style `setUp` and `tearDown` methods.
- **Exception Testing**: Use `pytest.raises` to assert that a specific exception is raised.
- **Parametrization**: Use `@pytest.mark.parametrize` to run the same test with different inputs, which helps in reducing code duplication and testing edge cases.

## 3. Production Code Integrity

A strict separation must be maintained between production code and test code.

- **No Test Code in Production**: Production files (anything under `esper/`) must **never** contain test-specific code, such as test harnesses, mock objects, or conditional logic for testing purposes.
- **Test Utilities**: Any test helper functions or utilities should be placed in the `tests/` directory.

## 4. System Philosophy: Fail Hard and Be Explicit

`esper` is a tightly integrated system. To ensure stability and prevent subtle bugs, we follow a "fail-hard" philosophy.

- **No Implicit Behavior**: Avoid "magic" or implicit behaviors. Code should be explicit and its intent should be clear.
- **Fail on Error**: It is always better for the system to fail with a `RuntimeException` or other clear error than to hide incorrect behavior behind default settings or silent error handling.
- **Explicit is Better than Implicit**: Do not rely on default values that could mask a problem. If a value is required for a component to function correctly, it should be required explicitly.
