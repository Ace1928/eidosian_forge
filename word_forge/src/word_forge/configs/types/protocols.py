"""
Protocol definitions for Word Forge configuration system.

This module provides Protocol definitions that define interfaces for
components that work with configuration and queue processing.

Classes:
    ConfigComponent: Protocol for configuration components with ENV_VARS
    JSONSerializable: Protocol for JSON-serializable objects
    QueueProcessor: Protocol for queue processing components
    WorkDistributor: Protocol for work distribution systems
    ConfigComponentInfo: Metadata about configuration components
"""

from dataclasses import dataclass, field
from typing import (
    Any,
    ClassVar,
    Dict,
    Protocol,
    Set,
    Tuple,
    Type,
    TypeVar,
    runtime_checkable,
)

try:
    from eidosian_core import eidosian
except ImportError:
    def eidosian():
        """Fallback decorator when eidosian_core is not available."""
        def decorator(func):
            return func
        return decorator

# Import from siblings
from .errors import Result
from .workers import TaskPriority

# Type variable for contravariant protocol
T_contra = TypeVar("T_contra", contravariant=True)

# Environment variable type
EnvVarType = Type[str] | Type[int] | Type[float] | Type[bool]


@runtime_checkable
class ConfigComponent(Protocol):
    """Protocol defining interface for all configuration components.

    All configuration components must implement this protocol to ensure
    consistency across the system, especially for environment variable
    overriding operations.

    Attributes:
        ENV_VARS: Class variable mapping environment variable names to
                 attribute names and their expected types for overriding
                 configuration values from environment.

    Example:
        ```python
        @dataclass
        class DatabaseConfig:
            db_path: str = "data/wordforge.db"
            pool_size: int = 5

            ENV_VARS: ClassVar[Dict[str, Tuple[str, EnvVarType]]] = {
                "WORDFORGE_DB_PATH": ("db_path", str),
                "WORDFORGE_DB_POOL_SIZE": ("pool_size", int),
            }
        ```
    """

    ENV_VARS: ClassVar[Dict[str, Tuple[str, EnvVarType]]]


@runtime_checkable
class JSONSerializable(Protocol):
    """Protocol for objects that can be serialized to JSON.

    Types implementing this protocol can be converted to JSON-compatible
    string representations for storage, transmission, or display purposes.

    Example:
        ```python
        class ConfigObject(JSONSerializable):
            def __init__(self, name: str, value: int):
                self.name = name
                self.value = value

            def __str__(self) -> str:
                return f"{{'name': '{self.name}', 'value': {self.value}}}"
        ```
    """

    def __str__(self) -> str:
        """Convert object to string representation for serialization.

        Returns:
            str: A string representation suitable for JSON serialization
        """
        ...


class QueueProcessor(Protocol[T_contra]):
    """
    Protocol defining the interface for queue processing components.

    Implementations define how items from a queue are processed,
    providing a consistent interface for worker threads.
    """

    @eidosian()
    def process(self, item: T_contra) -> Result[bool]:
        """
        Process an item from the queue.

        Args:
            item: The item to process

        Returns:
            Result indicating success or failure with context
        """
        ...


class WorkDistributor(Protocol):
    """
    Protocol for work distribution systems that manage task parallelism.

    Defines the interface for submitting tasks to a pool of worker threads
    and retrieving results, with support for priorities and backpressure.
    """

    @eidosian()
    def submit(
        self, task: Any, priority: TaskPriority = TaskPriority.NORMAL
    ) -> Result[Any]:
        """
        Submit a task for processing with the specified priority.

        Args:
            task: The task to process
            priority: The task priority

        Returns:
            Result containing the task result or error
        """
        ...

    @eidosian()
    def shutdown(self, wait: bool = True, cancel_pending: bool = False) -> None:
        """
        Shut down the work distributor.

        Args:
            wait: Whether to wait for pending tasks to complete
            cancel_pending: Whether to cancel pending tasks
        """
        ...


@dataclass(frozen=True)
class ConfigComponentInfo:
    """
    Metadata about a configuration component.

    Used to track component relationships and dependencies for reflection,
    dependency resolution, and runtime validation.

    Attributes:
        name: Component name used for registry lookup
        class_type: The class of the component for type checking
        dependencies: Names of other components this one depends on

    Example:
        ```python
        info = ConfigComponentInfo(
            name="database",
            class_type=DatabaseConfig,
            dependencies={"logging"}
        )
        ```
    """

    name: str
    class_type: Type[ConfigComponent]
    dependencies: Set[str] = field(default_factory=set)


__all__ = [
    "ConfigComponent",
    "JSONSerializable",
    "QueueProcessor",
    "WorkDistributor",
    "ConfigComponentInfo",
    "EnvVarType",
    "T_contra",
]
