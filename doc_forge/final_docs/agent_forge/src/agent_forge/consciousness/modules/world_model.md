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

`agent_forge/consciousness/modules/world_model.py` is a Python module that implements the world model module for an Eidosian system. This module plays a crucial role in understanding and predicting events within the agent's environment.

## API Documentation

### Functions

#### `WorldModelModule`

- **name**: `"world_model"`
  - Module name, unique identifier.
  
- **__init__()**
  - Initializes the world model module with default values.
    - `_seen_signatures`: A set to track seen event signatures.
    - `_transitions`: A dictionary to store transitions between events and their counts.
    - `_event_counts`: A counter to keep track of the number of events.
    - `_last_event_type`: The last observed event type, initialized as `None`.
    - `_belief_cache`: A cache for belief features.

- **rollout(steps=3, policy_hint=None)**
  - Rolls out the world model over a specified number of steps (default: 3).
    - Accepts optional parameters:
      - `steps`: Number of steps to roll out.
      - `policy_hint`: Optional hint about initial conditions.
    - Returns a list of dictionaries representing the state at each step.

- **_predict_transition(prev_event_type, current_event_type)**
  - Predicts the next event type based on previous and current event types.
    - Accepts two parameters:
      - `prev_event_type`: The last observed event type.
      - `current_event_type`: The current event type being evaluated.
    - Returns a dictionary containing the predicted transition, actual event type, probability of the actual event occurring, prediction error, known transition count.

- **tick(ctx)**
  - Updates the world model based on incoming tick events.
    - Accepts an `TickContext` object.
    - Tracks noise, perturbations, and error thresholds for predictions.
    - Updates the belief cache, calculates last prediction errors, and broadcasts events if necessary.

### Classes

#### `WorldModelModule`

- **name**: `"world_model"`
  - Unique identifier for the world model module.

- **rollout()**
  - Rolls out the world model over a specified number of steps.
    - Accepts parameters:
      - `steps`: Number of steps to roll out.
      - `policy_hint`: Optional hint about initial conditions.
    - Returns a list of dictionaries representing the state at each step.

- **_predict_transition()**
  - Predicts the next event type based on previous and current event types.
    - Accepts two parameters:
      - `prev_event_type`: The last observed event type.
      - `current_event_type`: The current event type being evaluated.
    - Returns a dictionary containing the predicted transition, actual event type, probability of the actual event occurring, prediction error, known transition count.

- **tick()**
  - Updates the world model based on incoming tick events.
    - Accepts an `TickContext` object.
    - Tracks noise, perturbations, and error thresholds for predictions.
    - Updates the belief cache, calculates last prediction errors, and broadcasts events if necessary.

### Parameters

[ Prompt: 59.5 t/s | Generation: 7.4 t/s ]

Exiting...