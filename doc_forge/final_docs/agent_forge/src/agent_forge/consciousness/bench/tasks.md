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

# File Summary: agent_forge/consciousness/bench/tasks.py

This file defines a set of trial tasks that can be used in cognitive experiments related to consciousness and attention. The `TrialTask` class is a dataclass that represents each task, with attributes such as the task name and description.

## API Documentation

### Available Tasks
The `_TASKS` dictionary contains available trial tasks along with their descriptions:

- **noop**: A no-op task for baseline/perturbation measurements.
- **signal_pulse**: Inject deterministic synthetic signal events to exercise attention/competition paths.
- **self_other_discrimination**: Inject controllability probes that alternate between self-caused and externally mimicked outcomes.
- **continuity_distraction**: Inject low-salience distractors around stable thread anchors to test working-set continuity under noise.
- **report_grounding_challenge**: Inject evidence probes that should be cited by grounded reports and flagged when unsupported.

### Methods

1. **available_tasks()**
   - Returns a sorted list of `TrialTask` objects, ordered by their names.

2. **resolve_task(name: str | None)**
   - Retrieves the corresponding `TrialTask` object based on the provided task name or "noop" if no name is provided.
   - If the provided name is not found in `_TASKS`, it returns the default "noop" task.

3. **apply_task_stage(**kwargs)**
   - Applies a specific stage to a given task within a state directory, using various parameters such as `state_dir`, `task`, `stage`, and `beat`.
   - The function handles different tasks based on their names:
     - For "signal_pulse", it applies the stage (baseline or perturb) with appropriate event strength.
     - For "self_other_discrimination", it performs self-other discrimination probes, considering the current stage and beat number.
     - For "continuity_distraction", it injects low-salience distractors around stable thread anchors, adjusting their strengths based on the task's stage.
     - For "report_grounding_challenge", it generates evidence probing events that are supported or unsupported by ground truth.

## Current Status

- This file is currently functional and provides a comprehensive set of tasks for cognitive experiments related to consciousness and attention.
  
## Potential Future Directions

1. **Enhancements**
   - Adding more detailed event descriptions could improve the clarity of the experiment reports.
   - Implementing additional task stages or variations on existing ones might further enhance the complexity and realism of the experimental setup.

2. **Refactoring**
   - Consolidating similar functions, such as `apply_task_stage`, into a single function with different parameters for each task stage would make the code more modular and easier to manage.
   - Refactoring event handling logic could lead to better organization and maintainability, especially if future enhancements require modifying specific stages.

[ Prompt: 33.6 t/s | Generation: 4.2 t/s ]

Exiting...