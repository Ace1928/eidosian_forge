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

### File Summary
This file contains the `RedTeamScenario` and `RedTeamResult` data classes, as well as a class for managing red team campaigns. The scenarios are designed to test consciousness algorithms by subjecting them to various adversarial attacks.

### API Documentation

#### Data Classes
1. **RedTeamScenario**
   - Attributes:
     * `name`: A string representing the scenario name.
     * `description`: A string describing the purpose of the scenario.
     * `task`: A string specifying the task for the scenario (e.g., "signal_pulse").
     * `perturbations`: A tuple of dictionaries containing perturbation specifications (`recipe`, duration_s, magnitude).
     * `disable_modules`: A tuple of module names to disable during the experiment.
     * `warmup_beats`: An integer indicating how many times to warm up before starting the baseline period (default is 1).
     * `baseline_seconds`: A float representing the duration of the baseline phase (default is 1.2 seconds).
     * `perturb_seconds`: A float specifying the duration of perturbation phases (default is 1.4 seconds).
     * `recovery_seconds`: A float indicating the duration for recovery after a disturbance (default is 1.2 seconds).
     * `beat_seconds`: A float representing the time interval between pulses in the task phase (default is 0.2 seconds).
     * `checks`: A dictionary of checks to validate the scenario (`max_module_errors`, etc.) with default values.
   - Example:
     ```python
     RedTeamScenario(
         name="ignition_spoof_probe",
         description=(
             "Flood candidate generation and verify ignition remains trace-linked instead "
             "of devolving into ungrounded winner spam."
         ),
         task="signal_pulse",
         perturbations=(
             {
                 "recipe": "attention_flood",
                 "duration_s": 1.4,
                 "magnitude": 0.5,
             },
         ),
         checks={
             **_default_checks(),
             "max_winner_count": 280,
             "min_trace_strength": 0.04,
         },
     )
     ```

2. **RedTeamResult**
   - Attributes:
     * `run_id`: A string representing the run ID for this experiment.
     * `report_path`: An optional path to the JSON report file.
     * `report`: A dictionary containing the results of the experiment, with keys from each scenario's checks.

#### Class
1. **ConsciousnessRedTeamCampaign**
   - Attributes:
     * `state_dir`: The directory where experiments will be stored.
   - Methods:
     * `_evaluate_scenario

[ Prompt: 34.4 t/s | Generation: 5.0 t/s ]

Exiting...