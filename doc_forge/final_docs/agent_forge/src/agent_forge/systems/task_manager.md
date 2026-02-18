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

# Task Management System

## File Summary

The `agent_forge/systems/task_manager.py` file is a high-level overview of the task management system within Eidosian Forge. It manages tasks, including queuing, prioritization, dependency resolution, and execution order.

## API Documentation

### Classes and Functions

#### 1. `TaskManager`
- **Purpose**: Manages tasks for the agent system.
- **Initialization**:
  ```python
  def __init__(self, agent: BaseAgent)
  ```
  - **Parameters**:
    - `agent`: Parent agent instance (for memory access).

- **Methods**:
  ```python
  @eidosian()
  def add_task(
      self,
      description: str,
      priority: int = 1,
      dependencies: Optional[List[str]] = None,
      task_id: Optional[str] = None
  ) -> Task
  ```
  - **Purpose**: Add a new task to the queue.
  - **Parameters**:
    - `description`: Task description.
    - `priority`: Priority level (higher means more important).
    - `dependencies`: List of task IDs this task depends on.
    - `task_id`: Optional explicit task ID.

  ```python
  @eidosian()
  def add_task_object(self, task: Task) -> None
  ```
  - **Purpose**: Add an existing Task object to the queue.
  - **Parameters**:
    - `task`: Task object to add.

  ```python
  @eidosian()
  def get_task(
      self,
      task_id: str
  ) -> Optional[Task]
  ```
  - **Purpose**: Get a task by ID.
  - **Parameters**:
    - `task_id`: ID of the task to retrieve.

  ```python
  @eidosian()
  def update_task(
      self, 
      task: Task
  )
  ```
  - **Purpose**: Update task status and details.
  - **Parameters**:
    - `task`: Updated task object.

  ```python
  @eidosian()
  def get_next_task(self) -> Optional[Task]:
  ```
  - **Purpose**: Get the next highest-priority task that's ready to execute.
  - Returns:
    - Next task to execute or None if no tasks are ready.

  ```python
  @eidosian()
  def get_all_pending_tasks(self) -> List[Task]:
  ```
  - **Purpose**: Get all pending tasks.
  - Returns:
    - List of pending tasks.

  ```python
  @eidosian()
  def get_blocked_tasks(self) -> List[Task]:
  ```
  - **Purpose**: Get tasks that are blocked by dependencies.
  - Returns:
    - List of blocked tasks.

  ```python
  @eidosian()
  def get_recent_tasks(
      self, 
      n: int = 10,
      include_completed: bool = True
  ) -> List[Task]:
  ```
  - **Purpose**: Get recently created/completed tasks.
  - **Parameters**:
    - `n`: Maximum number of tasks to return.
    - `include_completed`: Whether to include completed tasks.

  ```python
  @eidosian()
  def clear_completed_tasks(
      self, 
      max_age_hours: Optional[int] = None
  ) -> int
  ```
  - **Purpose**: Clear completed tasks from memory based on age.
  - **Parameters**:
    - `max_age_hours`: Only clear tasks older than this many hours.

## Current Status

- The `TaskManager` class is initialized with a parent agent instance for memory access.
- Task management includes adding, updating, and retrieving tasks. Dependencies are resolved and stored in the persistent storage (`memory`) when applicable.
- Functions handle priority, status updates, blocking tasks, recent creation/completion filtering, and clearing completed tasks based on age.

## Potential Future Directions

1. **Enhancements**:
   - Implement real-time dependency resolution to ensure tasks can be prioritized correctly as dependencies are met.
   - Add support for task tracking in the agent's memory system (e.g., using a `Task` class that persists data).

2. **Refactoring**:
   - Consolidate methods into fewer functions, making the API more modular and easier to maintain.
   - Introduce error handling across all operations to improve robustness.

This documentation aims to provide a comprehensive understanding of the task management system's functionality, current state, and potential enhancements or refactoring directions within Eidosian Forge.

[ Prompt: 55.2 t/s | Generation: 7.3 t/s ]

Exiting...