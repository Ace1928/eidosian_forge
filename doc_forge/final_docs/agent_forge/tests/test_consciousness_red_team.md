Loading model... 


▄▄ ▄▄
██ ██
██ ██  ▀▀█▄ ███▄███▄  ▀▀█▄    ▄████ ████▄ ████▄
██ ██ ▄█▀██ ██ ██ ██ ▄█▀██    ██    ██ ██ ██ ██
██ ██ ▀█▄██ ██ ██ ██ ▀█▄██ ██ ▀████ ████▀ ████▀
                                    ██    ██
                                    ▀▀    ▀▀

build      : b8083-ae2d3f28a
model      : qwen2.5-1.5b-instruct-q5_k_m.gguf
modalities : text

available commands:
  /exit or Ctrl+C     stop or exit
  /regen              regenerate the last response
  /clear              clear the chat history
  /read               add a text file


> <|im_start|>system
You are an Eidosian Documentation Forge agent. Your task is to generate detailed, accurate, and structurally elegant documentation for the provided source file.
Follow these requirements:
1. File Summary: High-level overview of what the file does.
2. API Documentation: List all functions, classes, and their parameters/methods.
3. Current Status: Accurate assessment of the file's current state based on its content and comments.
4. Potential Future Directions: Eidosian ideas for ... (truncated)

# File Summary

`agent_forge/tests/test_consciousness_red_team.py` is a test file that focuses on the `ConsciousnessRedTeamCampaign` class, which is part of the Eidosian documentation forge. This file aims to verify the functionality and behavior of the red team campaign in various scenarios.

## API Documentation

### Functions

1. **test_red_team_campaign_runs_and_emits_event**
   - **Parameters**: 
     - `tmp_path: Path`
   - **Returns**: None
   - **Description**: Tests whether a red team campaign runs successfully and emits an event at the end.
   
2. **test_red_team_campaign_latest_reads_persisted_report**
   - **Parameters**: 
     - `tmp_path: Path`
   - **Returns**: None
   - **Description**: Tests if the latest run of a red team campaign reads the persisted report correctly from disk.

3. **test_red_team_campaign_applies_shared_overlay_and_disable_modules**
   - **Parameters**:
     - `monkeypatch, tmp_path, Path`: The monkeypatch is used to simulate behavior for testing purposes.
   - **Returns**: None
   - **Description**: Tests whether the campaign applies a shared overlay and disables specified modules.

### Classes

1. **ConsciousnessRedTeamCampaign**
  - **Parameters**:
    - `base: Path`
  - **Methods**:
    - `run(scenarios, persist=False, base_seed=123)`: Runs a red team campaign with specified scenarios.
    - `latest()`: Returns the latest run of the campaign if persisted.

## Current Status

The file contains basic tests for the functionality and behavior of the `ConsciousnessRedTeamCampaign` class. The tests ensure that:

- A red team campaign runs successfully with various scenarios.
- The campaign emits events at the end of its execution.
- The latest run reads the persisted report correctly from disk.
- Specified overlays are applied, and modules are disabled.

### Potential Future Directions

1. **Logging**: Implement logging to capture more detailed information during tests for easier debugging.
2. **Parameter Validation**: Add parameter validation checks to ensure that inputs meet expected criteria before proceeding with test cases.
3. **Error Handling**: Improve error handling in the monkeypatch function and other parts of the script to provide better feedback when errors occur.

By implementing these suggestions, future versions of this file can offer more robust testing capabilities and a safer development environment for developers working on Eidosian documentation forge projects.

[ Prompt: 53.1 t/s | Generation: 9.1 t/s ]

Exiting...