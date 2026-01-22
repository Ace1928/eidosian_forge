# AGENTS.md

These instructions apply to all work in this repository.

## Engineering standards
- Create tests for every new function and update tests for any modified behavior.
- Maintain strict, complete documentation for all public modules, classes, and functions.
- Keep the architecture modular, extensible, and clearly structured.
- Prioritize high-performance, efficient implementations; avoid unnecessary allocations and work.
- Use dataclasses and strict contracts (type hints, validation, pre/postconditions) wherever practical.

## Verification requirements
- After any change, run profiling, unit tests, and integration tests to verify no unintended side effects.
- If tooling is missing, add or document it before completing the change.
