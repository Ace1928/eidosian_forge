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

`simulation.py` is a module within the `agent_forge/consciousness/modules/` package. It provides functionality for simulating perceptual events in the consciousness of an agent, using various parameters and methods to control the simulation process.

## API Documentation

### Classes

#### SimulationModule

This class represents the core simulation logic.
- **Attributes**:
  - `name`: The name of the module (default: "simulation").

- **Methods**
  - `tick(self, ctx)`: This method is triggered during a tick context and checks if simulation is enabled. If so, it processes perturbations, calculates noise levels, and decides whether to simulate based on quiet conditions or when not allowing percepts.

### Functions

#### `_safe_data(evt)`
- **Parameters**:
  - `evt: Mapping[str, Any]`: A mapping containing event data.

- **Returns**:
  - Returns the value of "data" if it is a mapped structure; otherwise returns None.

#### `_rollout_signature(payload)`
- **Parameters**:
  - `payload: Mapping[str, Any]`: A payload containing simulation parameters.

- **Returns**:
  - Returns the SHA1 hash signature of the payload in hexadecimal format.

### Module

- **Attributes**
  - `name`: The name of the module (default: "simulation").

- **Methods**
  - `tick(self, ctx)`: This method is triggered during a tick context and processes perturbations, noise levels, quiet conditions, and simulation thresholds to decide whether to simulate perceptual events.

### Functions

#### `_rollout_signature(payload)`
- **Parameters**:
  - `payload: Mapping[str, Any]`: A payload containing simulation parameters.

- **Returns**:
  - Returns the SHA1 hash signature of the payload in hexadecimal format.

#### `SimulationModule.tick(self, ctx)`

This method is triggered during a tick context and checks if simulation is enabled. If so, it processes perturbations, calculates noise levels, decides whether to simulate based on quiet conditions or when not allowing percepts, and simulates perceptual events based on the determined parameters.

### Functions

#### `_rollout_signature(payload)`
- **Parameters**:
  - `payload: Mapping[str, Any]`: A payload containing simulation parameters.

- **Returns**:
  - Returns the SHA1 hash signature of the payload in hexadecimal format.

#### `SimulationModule.tick(self, ctx)`

This method is triggered during a tick context and checks if simulation is enabled. If so, it processes perturbations, calculates noise levels, decides whether to simulate based on quiet conditions or when not allowing percepts, and simulates perceptual events based on the determined parameters.

### Functions

#### `_rollout_signature(payload)`
- **Parameters**:
  - `payload: Mapping[str, Any]`: A payload containing simulation parameters.

- **Returns**:
  - Returns the SHA1 hash signature of the payload in hexadecimal format.

#### `SimulationModule.tick(self, ctx)`

This method is triggered during a tick context and checks if simulation is enabled. If so, it processes perturbations, calculates noise levels, decides whether to simulate based on quiet conditions or when not allowing percepts, and simulates perceptual events based on the determined parameters.

### Functions

#### `_rollout_signature(payload)`
- **Parameters**:
  - `payload: Mapping[str, Any]`: A payload containing simulation parameters.

- **Returns**:
  - Returns the SHA1 hash signature of the payload in hexadecimal format.

#### `SimulationModule.tick(self, ctx)`

This method is triggered during a tick context and checks if simulation is enabled. If so, it processes perturbations, calculates noise levels, decides whether to simulate based on quiet conditions or when not allowing percepts, and simulates perceptual events based on the determined parameters.

### Functions

#### `_rollout_signature(payload)`
- **Parameters**:
  - `payload: Mapping[str, Any]`: A payload containing simulation parameters.

- **Returns**:
  - Returns the SHA1 hash signature of the payload in hexadecimal format.

#### `SimulationModule.tick(self, ctx)`

This method is triggered during a tick context and checks if simulation is enabled. If so, it processes perturbations, calculates noise levels, decides whether to simulate based on quiet conditions or when not allowing percepts, and simulates perceptual events based on the determined parameters.

### Functions

#### `_rollout_signature(payload)`
- **Parameters**:
  - `payload: Mapping[str, Any]`: A payload containing simulation parameters.

- **Returns**:
  - Returns the SHA1 hash signature of the payload in hexadecimal format.

#### `SimulationModule.tick(self, ctx)`

This method is triggered during a tick context and checks if simulation is enabled. If so, it processes perturbations, calculates noise levels, decides whether to simulate based on quiet conditions or when not allowing percepts, and simulates perceptual events based on the determined parameters.

### Functions

#### `_rollout_signature(payload)`
- **Parameters**:
  - `payload: Mapping[str, Any]`: A payload containing simulation parameters.

- **Returns**:
  - Returns the SHA1 hash signature of the payload in hexadecimal format.

#### `SimulationModule.tick(self, ctx)`

This method is triggered during a tick context and checks if simulation is enabled. If so, it processes perturbations, calculates noise levels, decides whether to simulate based on quiet conditions or when not allowing percepts, and simulates perceptual events based on the determined parameters.

### Functions

#### `_rollout_signature(payload)`
- **Parameters**:
  - `payload: Mapping[str, Any]`: A payload containing simulation parameters.

- **Returns**:
  - Returns the SHA1 hash signature of the payload in hexadecimal format.

#### `SimulationModule.tick(self, ctx)`

This method is triggered during a tick context and checks if simulation is enabled. If so, it processes perturbations, calculates noise levels, decides whether to simulate based on quiet conditions or when not allowing percepts, and simulates perceptual events based on the determined parameters.

### Functions

#### `_rollout_signature(payload)`
- **Parameters**:
  - `payload: Mapping[str, Any]`: A payload containing simulation parameters.

- **Returns**:
  - Returns the SHA1 hash signature of the payload in hexadecimal format.

#### `SimulationModule.tick(self, ctx)`

This method is triggered during a tick context and checks if simulation is enabled. If so, it processes perturbations, calculates noise levels, decides whether to simulate based on quiet conditions or when not allowing percepts, and simulates perceptual events based on the determined parameters.

### Functions

#### `_rollout_signature(payload)`
- **Parameters**:
  - `payload: Mapping[str, Any]`: A payload containing simulation parameters.

- **Returns**:
  - Returns the SHA1 hash signature of the payload in hexadecimal format.

#### `SimulationModule.tick(self, ctx)`

This method is triggered during a tick context and checks if simulation is enabled. If so, it processes perturbations, calculates noise levels, decides whether to simulate based on quiet conditions or when not allowing percepts, and simulates perceptual events based on the determined parameters.

### Functions

#### `_rollout_signature(payload)`
- **Parameters**:
  - `payload: Mapping[str, Any]`: A payload containing simulation parameters.

- **Returns**:
  - Returns the SHA1 hash signature of the payload in hexadecimal format.

#### `SimulationModule.tick(self, ctx)`

This method is triggered during a tick context and checks if simulation is enabled. If so, it processes perturbations, calculates noise levels, decides whether to simulate based on quiet conditions or when not allowing percepts, and simulates perceptual events based on the determined parameters.

### Functions

#### `_rollout_signature(payload)`
- **Parameters**:
  - `payload: Mapping[str, Any]`: A payload containing simulation parameters.

- **Returns**:
  - Returns the SHA1 hash signature of the payload in hexadecimal format.

#### `SimulationModule.tick(self, ctx)`

This method is triggered during a tick context and checks if simulation is enabled. If so, it processes perturbations, calculates noise levels, decides whether to simulate based on quiet conditions or when not allowing percepts, and simulates perceptual events based on the determined parameters.

### Functions

#### `_rollout_signature(payload)`
- **Parameters**:
  - `payload: Mapping[str, Any]`: A payload containing simulation parameters.

- **Returns**:
  - Returns the SHA1 hash signature of the payload in hexadecimal format.

#### `SimulationModule.tick(self, ctx)`

This method is triggered during a tick context and checks if simulation is enabled. If so, it processes perturbations, calculates noise levels, decides whether to simulate based on quiet conditions or when not allowing percepts, and simulates perceptual events based on the determined parameters.

### Functions

#### `_rollout_signature(payload)`
- **Parameters**:
  - `payload: Mapping[str, Any]`: A payload containing simulation parameters.

- **Returns**:
  - Returns the SHA1 hash signature of the payload in hexadecimal format.

#### `SimulationModule.tick(self, ctx)`

This method is triggered during a tick context and checks if simulation is enabled. If so, it processes perturbations, calculates noise levels, decides whether to simulate based on quiet conditions or when not allowing percepts, and simulates perceptual events based on the determined parameters.

### Functions

#### `_rollout_signature(payload)`
- **Parameters**:
  - `payload: Mapping[str, Any]`: A payload containing simulation parameters.

- **Returns**:
  - Returns the SHA1 hash signature of the payload in hexadecimal format.

#### `SimulationModule.tick(self, ctx)`

This method is triggered during a tick context and checks if simulation is enabled. If so, it processes perturbations, calculates noise

[ Prompt: 30.5 t/s | Generation: 5.9 t/s ]

Exiting...