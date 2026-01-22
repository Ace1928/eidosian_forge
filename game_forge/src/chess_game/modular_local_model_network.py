from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from queue import Queue
from lark import logger
from tenacity import retry, wait_exponential, stop_after_attempt
from transformers import (  # type: ignore[import]
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from accelerate import disk_offload  # type: ignore[import]
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    Tuple,
    Type,
    runtime_checkable,
    Deque,
)
from urllib.parse import urljoin
import uuid

import accelerate  # type: ignore[import]
import asyncio
import cProfile
import hashlib
import httpx
import importlib
import importlib.util
import inspect
import json
import logging
import os
import psutil
import queue
import re
import requests
import sys
import threading
import time
import torch
from weakref import WeakValueDictionary
from collections import deque
import os
from collections import deque
from typing import Dict, Any, List
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import hashlib
import requests
from tika import (  # type: ignore[import]
    parser,
)  # Keeping this import as it might be used by tika library internally, even if not directly used here.
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup  # type: ignore[import]
import logging
import asyncio
from cachetools import (  # type: ignore[import]
    LRUCache,
)

logging.basicConfig(level=logging.INFO)

# Constants
SEARXNG_URL = "http://localhost:8888"
TIKA_URL = "http://localhost:9998"
DOCUMENT_STORE = "./document_store"
PROCESSED_LOG = "processed_urls.json"

# Ensure document store exists - let's do this in __init__ or when the class is first used if needed.
os.makedirs(DOCUMENT_STORE, exist_ok=True)

## Interface Definitions
## Core Model Interfaces


class IModel(ABC):
    """
    **Abstract Interface: Model**

    Defines the base interface for all model classes within the system.

    **Purpose:**
        Ensures that all concrete model implementations provide a consistent
        :meth:`predict` method. This promotes polymorphism, allowing for
        interchangeable model components throughout the application.

    **Contracts:**
        - All concrete model classes MUST inherit from :class:`IModel`.
        - Subclasses MUST implement the :meth:`predict` method.
    """

    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """
        **Abstract Method: Predict**

        Defines the prediction method that all concrete model classes must implement.

        **Purpose:**
            To provide a standardized way to make predictions using different models.

        :param input_data: The input data for the model. The type is intentionally
                           flexible (:py:data:`~typing.Any`) to accommodate various
                           input formats (e.g., numpy arrays, dictionaries, custom objects).
        :type input_data: Any
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :return: The prediction output from the model. The output type is also
                 flexible (:py:data:`~typing.Any`) to support diverse prediction formats
                 (e.g., class labels, probabilities, numerical values).
        :rtype: Any
        """
        raise NotImplementedError


## Data Storage Interfaces


class IVectorStore(ABC):
    """
    **Abstract Interface: Vector Store**

    Defines the interface for interacting with vector databases or similar systems.

    **Purpose:**
        To abstract the underlying vector storage mechanism, allowing for
        interchangeable vector store implementations. This interface focuses on
        similarity searching capabilities.

    **Contracts:**
        - Concrete vector store classes MUST inherit from :class:`IVectorStore`.
        - Subclasses MUST implement the :meth:`similarity_search` method.
    """

    @abstractmethod
    def similarity_search(self, query: str) -> List[Any]:
        """
        **Abstract Method: Similarity Search**

        Defines the method for performing a similarity search within the vector store.

        **Purpose:**
            To find items in the vector store that are semantically similar to a given query.

        :param query: The query string for similarity search. This is the text used
                      to find similar items in the vector store.
        :type query: str
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :return: A list of results from the similarity search. The type and format
                 of each result are implementation-dependent (e.g., document IDs,
                 document objects, or other relevant data).
        :rtype: List[Any]
        """
        raise NotImplementedError


## Sub-module Interfaces


class ISubModule(ABC):
    """
    **Abstract Interface: Sub-module**

    Defines the interface for sub-modules within the system, focusing on asynchronous task processing.

    **Purpose:**
        To create independent, reusable components that can perform specific parts
        of a larger task asynchronously. This promotes modularity and concurrency.

    **Contracts:**
        - Concrete sub-module classes MUST inherit from :class:`ISubModule`.
        - Subclasses MUST implement the :meth:`process` method to handle tasks asynchronously.
    """

    @abstractmethod
    async def process(self, task: Any) -> Any:
        """
        **Abstract Method: Process (Asynchronous)**

        Defines the asynchronous method for processing a task within a sub-module.

        **Purpose:**
            To enable sub-modules to perform operations concurrently and non-blocking.

        :param task: The task object to be processed. The nature of the task is
                     flexible and can be defined by the specific sub-module.
        :type task: Any
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :return: The result of the task processing. The type of result is flexible
                 (:py:data:`~typing.Any`) to accommodate various processing outcomes.
        :rtype: Any
        """
        raise NotImplementedError


## Plugin Interfaces


class PluginBase(ABC):
    """
    **Abstract Base Class: Plugin**

    Base class for all plugins within the system.

    **Purpose:**
        To provide a consistent interface for plugins, enabling dynamic discovery
        and registration of plugin functionalities.

    **Contracts:**
        - All plugin classes MUST inherit from :class:`PluginBase`.
        - Subclasses MUST implement the :meth:`register_hooks` method.
    """

    @abstractmethod
    def register_hooks(self) -> None:
        """
        **Abstract Method: Register Hooks**

        Defines the entry point for plugins to register their functionality with the system.

        **Purpose:**
            To allow plugins to define and register hooks, callbacks, or extensions.
            The specifics of what constitutes a 'hook' are defined by the system's
            plugin architecture.

        :raises NotImplementedError: If the method is not implemented in a subclass.
        :rtype: None
        """
        raise NotImplementedError


## User Interface (Message) Interfaces


class IMessageFormatter(ABC):
    """
    **Abstract Interface: Message Formatter**

    Defines the interface for components responsible for formatting messages.

    **Purpose:**
        To abstract message formatting logic, allowing for different formatting
        styles or templates to be applied to messages.

    **Contracts:**
        - Concrete message formatter classes MUST inherit from :class:`IMessageFormatter`.
        - Subclasses MUST implement the :meth:`format_message` method.
    """

    @abstractmethod
    def format_message(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        **Abstract Method: Format Message**

        Defines the method to format a given message.

        **Purpose:**
            To apply specific formatting rules or templates to a raw message string.

        :param message: Raw message content to be formatted.
        :type message: str
        :param context: Optional dictionary providing additional context for dynamic
                        formatting. This can include variables or data to be inserted
                        into the message template.
        :type context: Optional[Dict[str, Any]]
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :return: The formatted message string.
        :rtype: str
        """
        raise NotImplementedError


class IMessageColorer(ABC):
    """
    **Abstract Interface: Message Colorer**

    Defines the interface for components responsible for coloring or styling messages.

    **Purpose:**
        To enhance message readability by applying color or styling based on
        severity or context. This allows for visual differentiation of message types.

    **Contracts:**
        - Concrete message colorer classes MUST inherit from :class:`IMessageColorer`.
        - Subclasses MUST implement the :meth:`color_message` method.
    """

    @abstractmethod
    def color_message(self, message: str, severity: str = "info") -> str:
        """
        **Abstract Method: Color Message**

        Defines the method to apply color or styling to a message.

        **Purpose:**
            To style messages based on their severity level (e.g., 'info', 'warning', 'error')
            or other contextual factors, improving user experience and information clarity.

        :param message: Message string to be colored or styled.
        :type message: str
        :param severity: Message severity level (e.g., 'info', 'warning', 'error').
                         Defaults to 'info'. This parameter guides the coloring/styling logic.
        :type severity: str
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :return: The styled message string.
        :rtype: str
        """
        raise NotImplementedError


## System Monitoring Interfaces


class IResourceMonitor(ABC):
    """
    **Abstract Interface: Resource Monitor**

    Defines the interface for system resource monitoring components.

    **Purpose:**
        To abstract system resource monitoring, allowing for different monitoring
        implementations and data sources. This interface focuses on providing
        resource usage statistics.

    **Contracts:**
        - Concrete resource monitor classes MUST inherit from :class:`IResourceMonitor`.
        - Subclasses MUST implement :meth:`start_monitoring` and :meth:`get_usage` methods.
    """

    @abstractmethod
    def start_monitoring(self, interval: int = 5) -> None:
        """
        **Abstract Method: Start Monitoring**

        Defines the method to start continuous resource monitoring.

        **Purpose:**
            To initiate the monitoring of system resources at a specified interval.
            This method sets up the monitoring process but does not directly return usage data.

        :param interval: Monitoring interval in seconds. Defaults to 5 seconds.
                         Determines how frequently resource usage is checked.
        :type interval: int
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :rtype: None
        """
        raise NotImplementedError

    @abstractmethod
    async def get_usage(self) -> Dict[str, float]:  # Make method async in interface
        """
        **Abstract Method: Get Usage (Asynchronous)**

        Defines the asynchronous method to retrieve current resource usage statistics.

        **Purpose:**
            To get the current usage of system resources such as CPU, memory, and disk.
            This method provides a snapshot of resource utilization at the time of invocation.

        :raises NotImplementedError: If the method is not implemented in a subclass.
        :return: Dictionary containing resource usage percentages. Keys might include
                 'cpu', 'memory', 'disk', etc., with values as floating-point percentages.
        :rtype: Dict[str, float]
        """
        raise NotImplementedError


## Profiling Interfaces


class IProfiler(ABC):
    """
    **Abstract Interface: Profiler**

    Defines the interface for code profiling components.

    **Purpose:**
        To abstract code profiling, allowing for different profiling tools or
        techniques to be used. This interface focuses on starting and stopping
        profiling sessions and retrieving analyzed results.

    **Contracts:**
        - Concrete profiler classes MUST inherit from :class:`IProfiler`.
        - Subclasses MUST implement :meth:`start_profiling` and :meth:`stop_profiling` methods.
    """

    @abstractmethod
    def start_profiling(self, context: Dict[str, Any]) -> cProfile.Profile:
        """
        **Abstract Method: Start Profiling**

        Defines the method to start a new profiling session.

        **Purpose:**
            To initiate the profiling of code execution, often within a specific context
            (e.g., for a particular task or operation).

        :param context: Dictionary providing execution context information for profiling.
                        This can include task IDs, operation names, or any relevant data
                        to categorize or filter profiling results.
        :type context: Dict[str, Any]
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :return: Instance of ``cProfile.Profile`` to manage the profiling session.
                 This object is used to stop and retrieve results from the profiling session.
        :rtype: cProfile.Profile
        """
        raise NotImplementedError

    @abstractmethod
    def stop_profiling(
        self, profiler: cProfile.Profile, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        **Abstract Method: Stop Profiling**

        Defines the method to stop a profiling session and return analyzed results.

        **Purpose:**
            To finalize a profiling session, analyze the collected data, and return
            profiling statistics. This method processes the raw profiling data into
            meaningful insights.

        :param profiler: Active ``cProfile.Profile`` instance to stop and analyze.
                         This is the object returned by :meth:`start_profiling`.
        :type profiler: cProfile.Profile
        :param context: Dictionary containing context information for profiling, e.g.,
                        'task_id', 'sort_by'. This context helps in analyzing and
                        interpreting the profiling results.
        :type context: Dict[str, Any]
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :return: Dictionary containing analyzed profiling statistics. The structure
                 and content of this dictionary are implementation-dependent but should
                 provide useful profiling information.
        :rtype: Dict[str, Any]
        """
        raise NotImplementedError


## Telemetry Interfaces


class ITelemetryCollector(ABC):
    """
    **Abstract Interface: Telemetry Collector**

    Defines the interface for telemetry data collection components.

    **Purpose:**
        To abstract telemetry data collection, allowing for different telemetry
        backends or services to be used. This interface focuses on recording metrics,
        capturing events, and managing spans for distributed tracing.

    **Contracts:**
        - Concrete telemetry collector classes MUST inherit from :class:`ITelemetryCollector`.
        - Subclasses MUST implement :meth:`record_metric`, :meth:`capture_event`,
          :meth:`start_span`, and :meth:`end_span` methods.
    """

    @abstractmethod
    def record_metric(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        **Abstract Method: Record Metric**

        Defines the method to record a telemetry metric.

        **Purpose:**
            To record numerical metrics, optionally with associated tags for categorization.
            Metrics are quantitative measurements of system behavior over time.

        :param name: Name of the metric. This should be a descriptive name that clearly
                     identifies the metric being recorded (e.g., 'requests_per_second', 'memory_usage').
        :type name: str
        :param value: Numeric value of the metric. This is the actual measurement being recorded.
        :type value: float
        :param tags: Optional dictionary of key-value tags for metric categorization.
                     Tags allow for filtering and aggregation of metrics based on dimensions
                     like service name, environment, or operation type. Defaults to None.
        :type tags: Optional[Dict[str, str]]
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :rtype: None
        """
        raise NotImplementedError

    @abstractmethod
    def capture_event(
        self, event_name: str, properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        **Abstract Method: Capture Event**

        Defines the method to capture a system event.

        **Purpose:**
            To record discrete events within the system, optionally with associated properties.
            Events are qualitative occurrences that happen at a specific point in time.

        :param event_name: Name or type of the event. This should be a descriptive name
                           that clearly identifies the event (e.g., 'user_login', 'error_occurred').
        :type event_name: str
        :param properties: Optional dictionary of event properties providing additional context.
                           Properties can include details like user ID, error message, or request parameters.
                           Defaults to None.
        :type properties: Optional[Dict[str, Any]]
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :rtype: None
        """
        raise NotImplementedError

    @abstractmethod
    def start_span(
        self,
        task_id: str,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        **Abstract Method: Start Span**

        Starts a telemetry span for a given task and operation.

        **Purpose:**
            To initialize a new span in the telemetry backend for distributed tracing.
            Spans track the duration and context of operations within a task, enabling
            performance monitoring and distributed tracing across services.

        :param task_id: Unique identifier for the task being tracked. This ID is used to
                        correlate spans belonging to the same task.
        :type task_id: str
        :param operation_name: A descriptive name for the operation being traced within the span.
                               This name should clearly indicate what operation is being performed
                               (e.g., 'database_query', 'http_request', 'process_data').
        :type operation_name: str
        :param attributes: Optional attributes to attach to the span at start, providing
                           initial context. Attributes are key-value pairs that add metadata
                           to the span (e.g., {'http.method': 'GET', 'http.url': '/api/users'}).
                           Defaults to None.
        :type attributes: Optional[Dict[str, Any]]
        :raises NotImplementedError: If the method is not implemented in a concrete backend.
        :return: An object representing the started span. The type is backend-specific and
                 must be passed to :meth:`end_span` to finalize the span. This object
                 is used to manage the lifecycle of the span.
        :rtype: Any
        """
        raise NotImplementedError

    @abstractmethod
    def end_span(self, span: Any, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        **Abstract Method: End Span**

        Ends a telemetry span.

        **Purpose:**
            To finalize a telemetry span, recording its duration and any end-span attributes.
            This marks the completion of the traced operation and sends the span data
            to the telemetry backend.

        :param span: The span object to end, as returned by :meth:`start_span`. This is the
                     object that represents the active span and is needed to finalize it.
        :type span: Any
        :param attributes: Optional attributes to attach to the span at end, providing
                           final context. These attributes are recorded when the span ends and
                           can include information like the operation's status or result.
                           Defaults to None.
        :type attributes: Optional[Dict[str, Any]]
        :raises NotImplementedError: If the method is not implemented in a concrete backend.
        :rtype: None
        """
        raise NotImplementedError


## Error Handling Interfaces


class IErrorHandler(ABC):
    """
    **Abstract Interface: Error Handler**

    Defines the interface for error handling components within the system.

    **Purpose:**
        To abstract error handling logic, allowing for different error reporting,
        logging, and recovery strategies. This interface focuses on handling errors
        within submodules and providing structured error information.

    **Contracts:**
        - Concrete error handler classes MUST inherit from :class:`IErrorHandler`.
        - Subclasses MUST implement :meth:`handle_error` and :meth:`log_error` methods.
    """

    @abstractmethod
    async def handle_error(
        self,
        task_id: str,
        error: Exception,
        submodule_name: str,
        context: Dict[str, Any],
        severity: str = "error",
    ) -> Dict[str, Any]:
        """
        **Abstract Method: Handle Error (Asynchronous)**

        Defines the asynchronous method to handle and process system errors within a submodule.

        **Purpose:**
            To manage errors, potentially logging them, reporting them, or attempting recovery.
            This method is designed to extract relevant information from the error context
            to provide detailed and actionable error handling.

        :param task_id: Unique ID of the task that encountered the error. This ID helps
                        in tracing errors back to their originating task.
        :type task_id: str
        :param error: Exception instance representing the error. This is the actual error
                      object that was raised.
        :type error: Exception
        :param submodule_name: Name of the submodule where the error occurred. This helps
                               identify the component that failed.
        :type submodule_name: str
        :param context: Dictionary providing context information about where the error occurred.
                        This can include request parameters, input data, or any other relevant
                        details for debugging and recovery.
        :type context: Dict[str, Any]
        :param severity: Severity level of the error (e.g., 'error', 'critical', 'warning').
                         Defaults to 'error'. This parameter categorizes the error's importance.
        :type severity: str
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :return: Dictionary containing information about the handled error. This can include
                 error IDs, recovery actions taken, or any other relevant information.
        :rtype: Dict[str, Any]
        """
        raise NotImplementedError

    @abstractmethod
    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: str,
        submodule_name: str,
        task_id: str,
    ) -> str:
        """
        **Abstract Method: Log Error**

        Defines the method to log error details with structured context and severity.

        **Purpose:**
            To create a structured log entry for an error, including contextual information,
            and return a reference ID for the log entry. This method ensures consistent
            and detailed error logging across the system.

        :param error: Exception instance representing the error. This is the actual error
                      object that needs to be logged.
        :type error: Exception
        :param context: Dictionary providing context information about where the error occurred.
                        This is the same context information provided to :meth:`handle_error`.
        :type context: Dict[str, Any]
        :param severity: Severity level of the error (e.g., 'error', 'critical', 'warning').
                         This is the same severity level provided to :meth:`handle_error`.
        :type severity: str
        :param submodule_name: Name of the submodule where the error occurred.
                               This is the same submodule name provided to :meth:`handle_error`.
        :type submodule_name: str
        :param task_id: Unique ID of the task that encountered the error.
                        This is the same task ID provided to :meth:`handle_error`.
        :type task_id: str
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :return: Error reference ID, which can be used to track or correlate error logs.
                 This ID provides a unique identifier for the log entry.
        :rtype: str
        """
        raise NotImplementedError


## Enums


class MetricTypes(Enum):
    """
    Enumeration of supported metric types for telemetry.

    This :py:class:`~enum.Enum` defines the types of metrics that can be tracked
    by the telemetry system. Each type represents a different way of measuring
    and monitoring system behavior.

    .. py:attribute:: COUNTER

       A metric that represents a single numerical value that only ever goes up.
       Useful for tracking counts of events, requests, or errors.

    .. py:attribute:: GAUGE

       A metric that represents a single numerical value that can arbitrarily go up and down.
       Suitable for measuring current levels, such as memory usage, CPU load, or queue size.

    .. py:attribute:: HISTOGRAM

       A metric that samples observations (usually things like request durations or response sizes)
       and counts them in configurable buckets. Histograms provide insights into the distribution
       of values over time, allowing for the calculation of percentiles and averages.
    """

    COUNTER = 1
    GAUGE = 2
    HISTOGRAM = 3


## Interfaces

### Model Management Interfaces


class ModelOptimizerInterface(Protocol):
    """Interface for ModelOptimizer components.

    Defines the contract for classes that optimize machine learning models,
    potentially adjusting configurations for performance, efficiency, or cost.
    """

    @abstractmethod
    def optimize(self, model: Any, config: Dict[str, Any]) -> Any:
        """Optimize a model instance"""
        ...

    @abstractmethod
    async def apply(
        self, optimized_config: Dict[str, Any], context: Dict[str, Any]
    ) -> Any:
        """Applies optimized configurations to a model.

        This method takes an optimized configuration and applies it to the relevant model
        or system components. The optimization process might involve techniques like pruning,
        quantization, or architecture search.

        :param optimized_config: A dictionary containing the optimized configuration parameters.
                                 The structure of this dictionary is specific to the optimization
                                 strategy and the model being optimized.
        :type optimized_config: Dict[str, Any]
        :param context: The current context dictionary, which may contain information about the
                        model, task, or environment relevant to applying the optimization.
        :type context: Dict[str, Any]
        :return: The result of applying the optimization, which could be confirmation of success,
                 performance metrics, or the updated model itself. The specific type is
                 implementation-dependent.
        :rtype: Any
        """
        ...


class ModelPoolInterface(Protocol):
    """
    Abstract interface for the Model Pool.

    Defines methods for accessing and managing models, including retrieval, preloading,
    and releasing models back to the pool.
    """

    def get_adaptive_model(self, model_type: str) -> Any:
        """
        Retrieves an adaptive model instance based on the specified type.
        """
        raise NotImplementedError("get_adaptive_model method must be implemented.")

    def preload_models(self, model_types: List[str]) -> None:
        """
        Preloads models of specified types into the pool.
        """
        raise NotImplementedError("preload_models method must be implemented.")

    def release_model(self, model_type: str) -> None:
        """
        Releases a model back to the pool, making it available for other tasks.
        """
        raise NotImplementedError("release_model method must be implemented.")

    def get_optimizer(self) -> ModelOptimizerInterface:
        """Get model optimization component"""
        raise NotImplementedError


### Vector Store Interface


class VectorCacheInterface(Protocol):
    """
    Interface for VectorCache components.

    Defines the contract for classes that manage a cache of vectors,
    providing efficient access and storage for vector data. This cache
    is intended to optimize vector retrieval and reduce computational overhead.
    """

    def get_store(self) -> IVectorStore:
        """
        Retrieves the vector store instance.

        This method provides access to the underlying vector store, allowing
        operations such as vector similarity search, addition, and deletion.

        :return: An instance of :py:class:`IVectorStore`.
        :rtype: IVectorStore
        """
        ...


### Telemetry and Monitoring Interfaces


class TelemetryCollectorInterface(Protocol):
    """
    Interface for TelemetryCollector components.

    Defines the contract for classes that collect and manage telemetry data,
    such as spans and metrics, for monitoring and observability of the system.
    Implementations should support starting and ending spans, and recording
    various types of metrics.
    """

    def start_span(
        self,
        task_id: str,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Starts a telemetry span for a given task and operation.

        A span represents a unit of work within a task, allowing for detailed
        performance tracing and analysis. Spans can be nested to represent
        sub-operations.

        :param task_id: The ID of the task for which to start a span.
        :type task_id: str
        :param operation_name: A descriptive name for the operation being traced within the span.
        :type operation_name: str
        :param attributes: Optional attributes to attach to the span, providing contextual information. Defaults to None.
        :type attributes: Optional[Dict[str, Any]]
        :return: An object representing the span, which must be used to record events and end the span. The specific type is backend-dependent.
        :rtype: Any
        """
        ...

    def end_span(self, span: Any, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Ends a telemetry span.

        Completes the span and records its duration and any additional attributes.
        Ensure that every started span is eventually ended to avoid resource leaks
        and ensure accurate telemetry data.

        :param span: The span object to end, as returned by :meth:`start_span`.
        :type span: Any
        :param attributes: Optional attributes to add when ending the span, providing final contextual information or status. Defaults to None.
        :type attributes: Optional[Dict[str, Any]]
        :rtype: None
        """
        ...


@runtime_checkable
class ITracingBackend(Protocol):
    """
    Abstract interface for tracing backends used by :class:`Tracing`.

    Defines methods for logging trace messages. Implementations should handle
    the actual logging or exporting of trace data to various tracing systems
    (e.g., Jaeger, Zipkin, in-memory). This abstraction allows for switching
    tracing backends without modifying the core tracing logic.
    """

    @abstractmethod
    def trace(
        self, task_id: str, message: str, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Abstract method to log a trace message.

        This method should be implemented by concrete tracing backends to
        handle the storage or transmission of trace messages. It provides a
        consistent interface for logging trace events throughout the application.

        :param task_id: The ID of the task being traced, providing context to the trace message.
        :type task_id: str
        :param message: The trace message to log. Should be a descriptive string
                        about the event being traced.
        :type message: str
        :param attributes: Optional attributes to attach to the trace message, providing
                           additional contextual information. Defaults to None.
        :type attributes: Optional[Dict[str, Any]]
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :rtype: None
        """
        raise NotImplementedError


class ITelemetryBackend(Protocol):
    """
    Interface for telemetry backends used by :class:`TelemetryCollector`.

    Defines the contract for telemetry backends, allowing for different implementations
    such as in-memory storage, or integration with external telemetry systems like Jaeger or Zipkin.
    This interface ensures that the telemetry collection is decoupled from the specific
    backend implementation.
    """

    @abstractmethod
    def start_span(
        self,
        task_id: str,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Starts a telemetry span for a given task and operation.

        Initializes a new span in the telemetry backend. Spans are used to track the
        duration and context of operations within a task, enabling performance monitoring
        and distributed tracing.

        :param task_id: Unique identifier for the task being tracked.
        :type task_id: str
        :param operation_name: A descriptive name for the operation being traced within the span.
        :type operation_name: str
        :param attributes: Optional attributes to attach to the span at start, providing initial context. Defaults to None.
        :type attributes: Optional[Dict[str, Any]]
        :raises NotImplementedError: If the method is not implemented in a concrete backend.
        :return: An object representing the started span. The type is backend-specific and must be
                 passed to :meth:`end_span` to finalize the span.
        :rtype: Any
        """
        raise NotImplementedError

    @abstractmethod
    def end_span(self, span: Any, attributes: Optional[Dict[str, Any]] = None) -> None:
        """
        Ends a telemetry span.

        Finalizes and records the span in the telemetry backend. This typically involves
        calculating the span duration and storing it along with any associated attributes.

        :param span: The span object to end, as returned by :meth:`start_span`. This object
                     is backend-specific and represents the active span.
        :type span: Any
        :param attributes: Optional attributes to add when ending the span, providing final
                           contextual information or status. Defaults to None.
        :type attributes: Optional[Dict[str, Any]]
        :raises NotImplementedError: If the method is not implemented in a concrete backend.
        :rtype: None
        """
        raise NotImplementedError

    @abstractmethod
    def record_metric(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def capture_event(
        self, event_name: str, properties: Optional[Dict[str, Any]] = None
    ) -> None:
        raise NotImplementedError


@runtime_checkable
class IMetricsBackend(Protocol):
    """
    Abstract interface for metrics backends used by :class:`MetricsRegistry`.

    Defines methods for creating, incrementing, setting, and observing metrics.
    This interface ensures that different metrics backends (e.g., in-memory,
    Prometheus, StatsD) can be used interchangeably with the :class:`MetricsRegistry`,
    allowing for flexible monitoring and metrics export.
    """

    @abstractmethod
    def create_counter(
        self,
        name: str,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Abstract method to create a counter metric.

        Counters are metrics that represent a single numerical value that only ever goes up.
        Useful for tracking counts of events, requests, or errors.

        :param name: The name of the counter metric. Must be unique within the registry.
        :type name: str
        :param description: A human-readable description of the metric. Defaults to None.
        :type description: Optional[str]
        :param labels: Optional labels (dimensions) to attach to the metric. These labels allow
                       for slicing and dicing of metric data. Defaults to None.
        :type labels: Optional[Dict[str, str]]
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :return: The created counter metric object. The exact type is backend-specific and will
                 be used in subsequent operations like incrementing.
        :rtype: Any
        """
        raise NotImplementedError

    @abstractmethod
    def create_gauge(
        self,
        name: str,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Abstract method to create a gauge metric.

        Gauges are metrics that represent a single numerical value that can arbitrarily go up and down.
        Suitable for measuring current levels, such as memory usage, CPU load, or queue size.

        :param name: The name of the gauge metric. Must be unique within the registry.
        :type name: str
        :param description: A human-readable description of the metric. Defaults to None.
        :type description: Optional[str]
        :param labels: Optional labels (dimensions) to attach to the metric. These labels allow
                       for slicing and dicing of metric data. Defaults to None.
        :type labels: Optional[Dict[str, str]]
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :return: The created gauge metric object. The exact type is backend-specific and will
                 be used in subsequent operations like setting the value.
        :rtype: Any
        """
        raise NotImplementedError

    @abstractmethod
    def create_histogram(
        self,
        name: str,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Any:
        """
        Abstract method to create a histogram metric.

        Histograms sample observations (usually things like request durations or response sizes) and count them in configurable buckets.
        Histograms provide insights into the distribution of values over time, allowing for the calculation of percentiles and averages.

        :param name: The name of the histogram metric. Must be unique within the registry.
        :type name: str
        :param description: A human-readable description of the metric. Defaults to None.
        :type description: Optional[str]
        :param labels: Optional labels (dimensions) to attach to the metric. These labels allow
                       for slicing and dicing of metric data. Defaults to None.
        :type labels: Optional[Dict[str, str]]
        :param buckets: Custom bucket boundaries for the histogram. If None, default buckets will be used.
                        Buckets should be in ascending order. Defaults to None.
        :type buckets: Optional[List[float]]
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :return: The created histogram metric object. The exact type is backend-specific and will
                 be used in subsequent operations like observing values.
        :rtype: Any
        """
        raise NotImplementedError

    @abstractmethod
    def increment_counter(
        self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Abstract method to increment a counter metric.

        Increases the value of the named counter by the given value.

        :param name: The name of the counter metric. Must correspond to a counter created with :meth:`create_counter`.
        :type name: str
        :param value: The value to increment by. Must be a positive integer. Defaults to 1.
        :type value: int
        :param labels: Optional labels to apply when incrementing. Must match the labels defined at creation
                       for the counter. Defaults to None.
        :type labels: Optional[Dict[str, str]]
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :raises ValueError: If the metric name is invalid or labels are incorrect.
        :rtype: None
        """
        raise NotImplementedError

    @abstractmethod
    def set_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Abstract method to set the value of a gauge metric.

        Sets the gauge metric to the specified value.

        :param name: The name of the gauge metric. Must correspond to a gauge created with :meth:`create_gauge`.
        :type name: str
        :param value: The numerical value to set the gauge to.
        :type value: float
        :param labels: Optional labels to apply when setting the gauge. Must match the labels defined at creation
                       for the gauge. Defaults to None.
        :type labels: Optional[Dict[str, str]]
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :raises ValueError: If the metric name is invalid or labels are incorrect.
        :rtype: None
        """
        raise NotImplementedError

    @abstractmethod
    def observe_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Abstract method to observe a value in a histogram metric.

        Records a single observation in the histogram. The value will be added to the appropriate bucket.

        :param name: The name of the histogram metric. Must correspond to a histogram created with :meth:`create_histogram`.
        :type name: str
        :param value: The value to observe.
        :type value: float
        :param labels: Optional labels to apply when observing. Must match the labels defined at creation
                       for the histogram. Defaults to None.
        :type labels: Optional[Dict[str, str]]
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :raises ValueError: If the metric name is invalid or labels are incorrect.
        :rtype: None
        """
        raise NotImplementedError


### Task Processing and Pipeline Interfaces


class TaskPipelineInterface(Protocol):
    """
    Interface for TaskPipeline components.

    Defines the contract for classes that orchestrate the execution of a sequence of tasks
    or sub-modules in a defined pipeline. Pipelines manage the flow of data and control
    between different processing stages.
    """

    @abstractmethod
    async def execute(self, pipeline: list, task: "Task") -> Dict[str, Any]:
        """
        Executes a specific pipeline of processing stages on a given task.

        This method defines the contract for executing a sequence of operations (the pipeline)
        on a given task. Implementations should process the task through each stage of the
        pipeline, potentially modifying the task or its context at each step.

        :param pipeline: A list of processing stages (e.g., functions, sub-modules) to be executed in order.
                       Each stage should accept and process the task.
        :type pipeline: list
        :param task: The task object to be processed through the pipeline.
        :type task: Task
        :return: A dictionary containing the results of the pipeline execution. The structure
                 and content of this dictionary are pipeline-specific.
        :rtype: Dict[str, Any]
        :raises NotImplementedError: If the method is not implemented by a concrete subclass.
        """
        raise NotImplementedError

    @abstractmethod
    async def process(self, task: "Task") -> "Task":
        """
        Processes a task through the entire task pipeline.

        This method defines the contract for initiating and managing the complete processing
        of a task from start to finish within the pipeline framework. It may involve selecting
        a pipeline, executing it, and handling the overall task lifecycle.

        :param task: The task object to be processed.
        :type task: Task
        :return: The processed task object, potentially modified or enriched by the pipeline execution.
        :rtype: Task
        :raises NotImplementedError: If the method is not implemented by a concrete subclass.
        """
        raise NotImplementedError


class SubModuleInterface(Protocol):
    """
    Interface for all SubModules.

    Defines the contract that all concrete submodules must implement,
    ensuring a consistent processing interface across different types of submodules.
    This interface promotes modularity and interchangeability of submodules within the system.
    """

    async def process(self, task: "Task") -> Any:
        """
        Processes a given task.

        This is the main entry point for task processing within a submodule. Concrete submodules
        must implement this method to perform their specific processing logic on the input task.

        :param task: The task to be processed. The task object encapsulates all necessary information
                     and data for the submodule to perform its operation.
        :type task: Task
        :return: The result of the task processing. The nature of the result is submodule-specific
                 and can be any type of data or object.
        :rtype: Any
        :raises NotImplementedError: If the method is not implemented by a concrete submodule.
        """
        ...


### Content Analysis Interfaces


class ContentExtractorInterface(Protocol):
    """
    Interface for content extraction modules.

    Defines the contract for classes that extract relevant content from a given context.
    Content extraction modules are responsible for identifying and retrieving specific
    pieces of information from unstructured or semi-structured data within the context.
    """

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the context to extract content.

        Analyzes the input context and extracts the desired content based on predefined rules
        or models. The extracted content is then added to the context for further processing.

        :param context: The input context dictionary. This dictionary may contain various forms of data
                        from which content needs to be extracted.
        :type context: Dict[str, Any]
        :return: The updated context with extracted content. The extracted content is typically
                 added as new key-value pairs in the context dictionary.
        :rtype: Dict[str, Any]
        """
        ...


class CodeAnalyzerInterface(Protocol):
    """
    Interface for code analysis modules.

    Defines the contract for classes that analyze code snippets within a context.
    Code analysis modules can perform tasks such as syntax checking, static analysis,
    security vulnerability detection, or code complexity assessment.
    """

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the context to analyze code snippets.

        Identifies and analyzes code snippets present in the input context. The analysis results,
        such as detected issues, metrics, or insights, are then incorporated back into the context.

        :param context: The input context dictionary. This dictionary may contain code snippets
                        in various formats or locations.
        :type context: Dict[str, Any]
        :return: The updated context with code analysis results. The results are typically added
                 as structured data within the context dictionary.
        :rtype: Dict[str, Any]
        """
        ...


class EmotionalAnalyzerInterface(Protocol):
    """
    Interface for emotional analysis modules.

    Defines the contract for classes that analyze emotional tone in text within a context.
    Emotional analysis modules are used to detect and classify emotions expressed in textual data,
    such as sentiment analysis, emotion recognition, or tone detection.
    """

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the context to analyze emotional tone.

        Examines textual data within the input context to determine the emotional tone or sentiment.
        The analysis results, such as emotion labels or sentiment scores, are then added to the context.

        :param context: The input context dictionary. This dictionary may contain text data in various
                        fields or formats.
        :type context: Dict[str, Any]
        :return: The updated context with emotional analysis results. The results are typically
                 represented as structured data within the context dictionary.
        :rtype: Dict[str, Any]
        """
        ...


class SummaryGeneratorInterface(Protocol):
    """
    Interface for summary generation modules.

    Defines the contract for classes that generate summaries of text within a context.
    Summary generation modules are used to condense large amounts of text into shorter,
    informative summaries, preserving the key information and main points.
    """

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the context to generate summaries.

        Identifies and processes textual data within the input context to create concise summaries.
        The generated summaries are then added to the context for further use or output.

        :param context: The input context dictionary. This dictionary may contain large volumes of
                        text data that need to be summarized.
        :type context: Dict[str, Any]
        :return: The updated context with summary results. The summaries are typically added as
                 new text fields within the context dictionary.
        :rtype: Dict[str, Any]
        """
        ...


class GapAnalyzerInterface(Protocol):
    """
    Interface for gap analysis modules.

    Defines the contract for classes that analyze gaps or discrepancies within a context.
    Gap analysis modules are used to identify missing information, inconsistencies, or
    discrepancies in data, knowledge, or processes within the given context.
    """

    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the context to analyze gaps or discrepancies.

        Examines the input context to identify gaps, inconsistencies, or missing elements based
        on predefined criteria or models. The analysis results, highlighting the identified gaps,
        are then added to the context.

        :param context: The input context dictionary. This dictionary may contain data, knowledge,
                        or process descriptions that need to be analyzed for gaps.
        :type context: Dict[str, Any]
        :return: The updated context with gap analysis results. The results typically describe
                 the identified gaps and their locations within the context.
        :rtype: Dict[str, Any]
        """
        ...


### Error Handling and Validation Interfaces


class ErrorHandlerInterface(Protocol):
    """
    Interface for ErrorHandler components.

    Defines the contract for classes that handle errors occurring during task processing,
    providing mechanisms for logging, recovery, or fallback. Error handlers ensure
    system resilience and graceful degradation in the face of unexpected issues.
    """

    async def handle_error(
        self,
        task_id: str,
        error: Exception,
        submodule_name: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Handles an error that occurred during submodule processing.

        Provides a centralized point for error management, allowing for actions such as
        logging the error, attempting recovery, applying fallback logic, or escalating
        the error handling to a higher level.

        :param task_id: The ID of the task that encountered the error.
        :type task_id: str
        :param error: The exception that was raised during task processing.
        :type error: Exception
        :param submodule_name: The name of the submodule where the error occurred. This helps in
                                pinpointing the source of the error.
        :type submodule_name: str
        :param context: The current context dictionary at the point of error. This can be used
                        for debugging or recovery purposes.
        :type context: Dict[str, Any]
        :return: A dictionary containing details about the error and any results from
                 fallback attempts.
        :rtype: Dict[str, Any]
        """
        ...

    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: str,
        submodule_name: str,
        task_id: str,
    ) -> str:
        """
        Logs the error with detailed context, severity, submodule, and task ID.

        This method generates a detailed error message and logs it using the configured
        logging level. It includes contextual information, severity, submodule name,
        and task ID to provide comprehensive error tracking.

        :param error: The exception to log.
        :type error: Exception
        :param context: Contextual information related to the error.
        :type context: Dict[str, Any]
        :param severity: Severity level of the error.
        :type severity: str
        :param submodule_name: The name of the submodule where the error occurred.
        :type submodule_name: str
        :param task_id: The unique identifier of the task that encountered the error.
        :type task_id: str
        :return: A string message describing the logged error.
        :rtype: str
        """
        ...


class ValidationEngineInterface(ABC):
    """
    Abstract interface for validation engines.

    Defines the contract for classes that perform output validation,
    emphasizing context-aware validation to ensure the quality and correctness of task outputs.
    """

    @abstractmethod
    def register_validator(
        self,
        validator_name: str,
        validator_func: Callable[[Any, str, Dict[str, Any]], bool],
    ) -> None:
        """
        Abstract method to register a validator function.

        :param validator_name: The name of the validator.
        :type validator_name: str
        :param validator_func: The validator function that accepts output, task ID, and context.
        :type validator_func: Callable[[Any, str, Dict[str, Any]], bool]
        """
        raise NotImplementedError

    @abstractmethod
    def validate(
        self,
        output: Any,
        task_id: str,
        context: Dict[str, Any],
        validator_names: Optional[List[str]] = None,
    ) -> bool:
        """
        Abstract method to validate output against registered validators, considering context.

        :param output: The output to be validated.
        :type output: Any
        :param task_id: The ID of the task producing the output.
        :type task_id: str
        :param context: Contextual information for validation.
        :type context: Dict[str, Any]
        :param validator_names: Optional list of validator names to apply.
        :type validator_names: Optional[List[str]]
        :return: True if validation passes, False otherwise.
        :rtype: bool
        """
        raise NotImplementedError


class FallbackHandlerInterface(Protocol):
    """
    Interface for FallbackHandler components.

    Defines the contract for classes that provide fallback mechanisms when primary
    processing paths fail or are unavailable, ensuring system resilience and continuity
    of service. Fallback handlers are invoked when errors occur or when primary systems are down.
    """

    async def handle_fallback(
        self, task_id: str, context: Dict[str, Any], strategy_name: Optional[str] = None
    ) -> Any:
        """
        Handles a fallback scenario for a given task.

        Executes predefined fallback logic, which might involve returning a default response,
        using a cached result, invoking an alternative service, or any other action that
        maintains system functionality in the absence of the primary processing path.

        :param task_id: The ID of the task for which fallback is being handled. This helps in
                        selecting the appropriate fallback strategy if needed.
        :type task_id: str
        :param context: The current context dictionary, which may contain information relevant
                        to the fallback decision or process.
        :type context: Dict[str, Any]
        :param strategy_name: Optional name of the fallback strategy to use.
        :type strategy_name: Optional[str]
        :return: The result of the fallback handling, which could be an alternative output,
                 a default value, or an action indicating successful fallback. The specific type
                 is implementation-dependent.
        :rtype: Any
        """
        ...


### Benchmark Interface


class BenchmarkSuiteInterface(Protocol):
    """Interface for BenchmarkSuite components.

    Defines the contract for classes that manage and execute benchmark suites
    to evaluate system performance, find optimal configurations, or compare
    different implementations. Benchmark suites provide a structured way to assess
    and improve system capabilities.
    """

    async def find_optimal_config(
        self, task: "Task", context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Finds the optimal configuration for a given task using benchmarking.

        Executes a series of benchmarks to explore different configurations and identify
        the one that yields the best performance according to defined metrics. This process
        might involve automated experimentation and analysis of benchmark results.

        :param task: The task for which to find the optimal configuration. The task object
                     encapsulates the workload and requirements for benchmarking.
        :type task: Task
        :param context: The current context dictionary, which may contain information about
                        the environment, available resources, or specific benchmarking parameters.
        :type context: Dict[str, Any]
        :return: A dictionary representing the optimal configuration found for the given task.
                 The structure of this dictionary is specific to the system and benchmarking process.
        :rtype: Dict[str, Any]
        """
        ...


### Persistence Interface


class IPersistenceBackend(Protocol):
    """
    Abstract interface for persistence backends used by :class:`StateManager`.

    Defines methods for saving and loading state data. Implementations should
    handle the actual persistence mechanism, such as writing to disk, database,
    or in-memory storage. This abstraction allows for different persistence strategies
    to be used without affecting the core state management logic.
    """

    @abstractmethod
    def load_state(self) -> Dict[str, Any]:
        """
        Abstract method to load state data.

        Implementations should retrieve the state data from the persistence layer.
        If no state exists, it should return an empty dictionary to represent
        a fresh state. This method is called to initialize or restore the application state.

        :raises NotImplementedError: If the method is not implemented in a subclass.
        :return: The loaded state data. If no state was previously saved, an empty dictionary
                 should be returned.
        :rtype: Dict[str, Any]
        """
        raise NotImplementedError

    @abstractmethod
    def save_state(
        self, state_data: Dict[str, Any], task_id: Optional[str] = None
    ) -> None:
        """
        Abstract method to save state data.

        Implementations should persist the provided state data to the persistence
        layer. This method is called to update and store the application state.

        :param state_data: The state data to be saved. This is typically a dictionary
                           containing the application's current state.
        :type state_data: Dict[str, Any]
        :param task_id: Optional task ID associated with this state saving operation. This can be
                        useful for tracking or auditing state changes. Defaults to None.
        :type task_id: Optional[str]
        :raises NotImplementedError: If the method is not implemented in a subclass.
        :rtype: None
        """
        raise NotImplementedError


# Exceptions

## Configuration Exceptions


class ConfigError(Exception):
    """
    **Base Exception: Configuration Error**

    Base class for configuration related exceptions.

    **Purpose:**
        This exception serves as a general category for all configuration errors
        within the application. Specific configuration error types should subclass
        from this exception to provide more granular error handling.

    **Usage:**
        Subclass this exception for specific configuration error scenarios.

    :param message: A human-readable error message.
    :type message: str
    """

    def __init__(self, message: str):
        """
        Initialize a ConfigError.

        :param message: The error message.
        :type message: str
        """
        super().__init__(message)
        self.message = message


class ConfigLoadError(ConfigError):
    """
    **Configuration Exception: Load Error**

    Exception raised when the application fails to load configuration.

    **Purpose:**
        This error typically occurs during the startup phase of the application
        if the configuration file is missing, corrupted, or cannot be parsed.

    **Usage:**
        Raised by components responsible for loading application configuration.

    :param message: A human-readable error message detailing the load failure.
    :type message: str
    :param config_path: The path to the configuration file that failed to load. Defaults to None.
    :type config_path: Optional[str]
    """

    def __init__(self, message: str, config_path: Optional[str] = None):
        """
        Initialize a ConfigLoadError.

        :param message: The error message.
        :type message: str
        :param config_path: The path to the config file that failed to load (optional).
        :type config_path: Optional[str]
        """
        super().__init__(message)
        self.config_path = config_path


class ConfigValueError(ConfigError):
    """
    **Configuration Exception: Value Error**

    Exception raised when a specific configuration value is invalid.

    **Purpose:**
        This exception is used when the configuration file is loaded successfully,
        but one or more values within the configuration are not valid according to
        the application's requirements (e.g., incorrect type, out of range, invalid format).

    **Usage:**
        Raised when validating configuration values after successful loading.

    :param message: A human-readable error message describing the invalid value.
    :type message: str
    :param config_key: The configuration key that holds the invalid value. Defaults to None.
    :type config_key: Optional[str]
    :param config_value: The invalid configuration value itself. Defaults to None.
    :type config_value: Optional[Any]
    """

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[Any] = None,
    ):
        """
        Initialize a ConfigValueError.

        :param message: The error message.
        :type message: str
        :param config_key: The configuration key with the invalid value (optional).
        :type config_key: Optional[str]
        :param config_value: The invalid configuration value (optional).
        :type config_value: Optional[Any]
        """
        super().__init__(message)
        self.config_key = config_key
        self.config_value = config_value


## Task Exceptions


class InvalidTaskError(Exception):
    """
    **Task Exception: Invalid Task Error**

    Exception raised when a task is considered invalid or malformed.

    **Purpose:**
        This exception indicates that a task received by the system is not processable
        due to issues such as:

        - **Malformed Payload**: The task data structure is not as expected.
        - **Missing Required Information**: Essential fields or data are absent.
        - **Semantic Inconsistency**: The task request is logically flawed or contradictory.
        - **Unsupported Operation**: The task requests an action that is not supported in the current context.

    **Usage:**
        Raised by task processing components when an incoming task is deemed invalid.

    :param message: A detailed error message describing why the task is invalid.
    :type message: str
    :param task_payload: Optional payload of the invalid task, for debugging purposes. Defaults to None.
    :type task_payload: Optional[Any]
    """

    def __init__(self, message: str, task_payload: Optional[Any] = None):
        """
        Initialize an InvalidTaskError.

        :param message: The error message.
        :type message: str
        :param task_payload: The invalid task payload (optional).
        :type task_payload: Optional[Any]
        """
        super().__init__(message)
        self.message = message
        self.task_payload = task_payload


## Dependency Exceptions


class DependencyResolutionError(Exception):
    """
    **Dependency Exception: Resolution Error**

    Exception raised when dependency resolution fails.

    **Purpose:**
        This error occurs when the system is unable to resolve or satisfy
        a required dependency. This could be due to missing dependencies,
        conflicts between dependency versions, or issues in the dependency
        resolution mechanism itself.

    **Usage:**
        Raised by dependency injection or service locator components when
        a dependency cannot be resolved.

    :param message: A human-readable error message describing the dependency resolution failure.
    :type message: str
    :param dependency_name: Optional name of the dependency that could not be resolved. Defaults to None.
    :type dependency_name: Optional[str]
    :param context: Optional context information related to the dependency resolution attempt. Defaults to None.
    :type context: Optional[Dict[str, Any]]
    """

    def __init__(
        self,
        message: str,
        dependency_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a DependencyResolutionError.

        :param message: The error message.
        :type message: str
        :param dependency_name: The name of the dependency that failed to resolve (optional).
        :type dependency_name: Optional[str]
        :param context: Context information about the resolution failure (optional).
        :type context: Optional[Dict[str, Any]]
        """
        super().__init__(message)
        self.dependency_name = dependency_name
        self.context = context


# Configurations, Plugins, Features, Dependencies


class ConfigManager:
    """
    Manages application configuration with hot-reloading and listener support.

    Configuration is loaded from a JSON file and can be reloaded at runtime.
    Listeners can be registered to be notified of configuration changes,
    enabling dynamic updates throughout the application.

    :raises ConfigLoadError: If the configuration file cannot be loaded or parsed.
    :raises ConfigValueError: If there are issues with the configuration values.
    """

    def __init__(self, config_path: str = "config.json") -> None:
        """
        Initializes the ConfigManager.

        Loads the initial configuration from the specified JSON file and
        sets up the listener notification system.

        :param config_path: Path to the JSON configuration file. Defaults to 'config.json'.
        :type config_path: str
        :raises ConfigLoadError: If the configuration file is not found or is invalid JSON.
        """
        self.config_path = config_path
        """Path to the configuration file."""
        self.config: Dict[str, Any] = {}
        """Current configuration data, loaded from the config file."""
        self.listeners: List[Callable[[Dict[str, Any]], None]] = []
        """List of listener functions to be notified on config changes."""
        self._load_config()

    def _load_config(self) -> None:
        """
        Loads and parses the configuration from the JSON file.

        Handles file not found and JSON decoding errors, logging warnings or errors
        and setting the configuration to an empty dictionary in case of failure.

        :raises ConfigLoadError: If the configuration file is not found or is invalid JSON.
        """
        try:
            with open(self.config_path, "r") as f:
                self.config = json.load(f)
        except FileNotFoundError as e:
            logging.warning(
                f"Configuration file not found at {self.config_path}. Using default empty config. Error: {e}"
            )
            self.config = {}
            raise ConfigLoadError(
                f"Configuration file not found: {self.config_path}"
            ) from e
        except json.JSONDecodeError as e:
            logging.error(
                f"Error decoding JSON from {self.config_path}. Check config file format. Error: {e}"
            )
            self.config = {}
            raise ConfigLoadError(
                f"Invalid JSON in config file: {self.config_path}"
            ) from e
        except Exception as e:
            logging.error(
                f"Unexpected error loading config from {self.config_path}: {e}"
            )
            self.config = {}
            raise ConfigLoadError(
                f"Failed to load config from {self.config_path}"
            ) from e

    def get_config(self) -> Dict[str, Any]:
        """
        Returns the current configuration dictionary.

        :return: The current configuration.
        :rtype: Dict[str, Any]
        """
        return self.config

    def set_config(self, new_config: Dict[str, Any]) -> None:
        """
        Sets a new configuration and notifies all registered listeners.

        This method directly updates the current configuration with the provided
        dictionary and then triggers the notification process for all listeners.

        :param new_config: The new configuration dictionary to set.
        :type new_config: Dict[str, Any]
        :raises TypeError: If ``new_config`` is not a dictionary.
        """
        if not isinstance(new_config, dict):
            raise TypeError(f"new_config must be a dictionary, got {type(new_config)}")
        self.config = new_config
        self._notify_listeners()

    def add_listener(self, listener: Callable[[Dict[str, Any]], None]) -> None:
        """
        Adds a listener function to be notified when the configuration changes.

        Listener functions should accept a single argument, which is the
        configuration dictionary.

        :param listener: The listener function to add.
        :type listener: Callable[[Dict[str, Any]], None]
        :raises TypeError: If ``listener`` is not a callable.
        """
        if not callable(listener):
            raise TypeError(f"listener must be a callable, got {type(listener)}")
        self.listeners.append(listener)

    def _notify_listeners(self) -> None:
        """
        Notifies all registered listeners of the current configuration.

        Iterates through the list of registered listener functions and calls each
        one with the current configuration dictionary as an argument.
        """
        for listener in self.listeners:
            try:
                listener(self.config)
            except Exception as e:
                logging.error(f"Error notifying listener {listener}: {e}")

    def hot_reload_config(self) -> None:
        """
        Reloads the configuration from the file and notifies listeners.

        This method forces a reload of the configuration from the file path
        specified during initialization. After successfully reloading, it
        notifies all registered listeners about the configuration update.

        :raises ConfigLoadError: If the configuration file cannot be reloaded or parsed.
        """
        logging.info("Hot-reloading configuration...")
        try:
            self._load_config()
            self._notify_listeners()
            logging.info("Configuration hot-reloaded successfully.")
        except ConfigLoadError as e:
            logging.error(f"Configuration hot-reload failed: {e}")


class FeatureFlagManager:
    """
    Manages feature flags, allowing dynamic enabling/disabling of features.

    Feature flags are read from the configuration managed by a :class:`ConfigManager`.
    This allows for runtime control over application features without requiring
    code deployments.

    :param config_manager: The ConfigManager instance to retrieve configuration from.
    :type config_manager: ConfigManager
    :param feature_flag_prefix: Prefix in the configuration to identify feature flags.
                                 Defaults to 'feature_flags'.
    :type feature_flag_prefix: str, optional
    """

    def __init__(
        self, config_manager: ConfigManager, feature_flag_prefix: str = "feature_flags"
    ) -> None:
        """
        Initializes the FeatureFlagManager.

        :param config_manager: The ConfigManager instance to use for configuration.
        :type config_manager: ConfigManager
        :param feature_flag_prefix: Prefix for feature flags in the config.
        :type feature_flag_prefix: str, optional
        :raises TypeError: If ``config_manager`` is not a :class:`ConfigManager`.
        :raises ValueError: If ``feature_flag_prefix`` is empty.
        """
        if not isinstance(config_manager, ConfigManager):
            raise TypeError(
                f"config_manager must be an instance of ConfigManager, got {type(config_manager)}"
            )
        if not feature_flag_prefix:
            raise ValueError("feature_flag_prefix cannot be empty.")

        self.config_manager = config_manager
        """The ConfigManager instance used to fetch configurations."""
        self.feature_flag_prefix = feature_flag_prefix
        """Prefix used to identify feature flags in the configuration."""

    def is_enabled(self, flag_name: str) -> bool:
        """
        Checks if a feature flag is enabled.

        Looks up the feature flag in the configuration under the specified prefix.
        If the flag is not found, it defaults to disabled (``False``).

        :param flag_name: The name of the feature flag to check.
        :type flag_name: str
        :return: ``True`` if the feature flag is enabled, ``False`` otherwise.
        :rtype: bool
        :raises TypeError: If ``flag_name`` is not a string.
        :raises ValueError: If ``flag_name`` is empty.
        """
        if not isinstance(flag_name, str):
            raise TypeError(f"flag_name must be a string, got {type(flag_name)}")
        if not flag_name:
            raise ValueError("flag_name cannot be empty.")

        config = self.config_manager.get_config()
        feature_flags = config.get(self.feature_flag_prefix, {})
        return feature_flags.get(flag_name, False)


class PluginManager:
    """
    Dynamically loads and manages plugins for extending application functionality.

    Plugins are loaded from specified paths (directories or individual files)
    and are expected to be subclasses of :class:`PluginBase` with a
    :meth:`PluginBase.register_hooks` method.

    :param plugin_paths: List of paths to plugin directories or files. Defaults to an empty list.
    :type plugin_paths: Optional[List[str]], optional
    """

    def __init__(self, plugin_paths: Optional[List[str]] = None) -> None:
        """
        Initializes the PluginManager.

        :param plugin_paths: List of paths to plugin directories or files.
                             Each path can be a directory containing plugin modules
                             or a direct path to a plugin file (.py). Defaults to an empty list.
        :type plugin_paths: Optional[List[str]], optional
        """
        self.plugin_paths = plugin_paths or []
        """List of paths from which plugins are loaded."""
        self.plugins: List[PluginBase] = []
        """List of loaded plugin instances."""
        self._load_plugins()

    def _load_plugins(self) -> None:
        """
        Loads plugins from the configured plugin paths.

        Iterates through each path in ``plugin_paths``, attempting to load
        Python modules from directories or directly from files. Modules are
        inspected for subclasses of :class:`PluginBase`, which are then
        instantiated and added to the list of loaded plugins.

        :raises ImportError: If there are issues importing plugin modules.
        :raises Exception: For any other errors during plugin loading or instantiation.
        """
        for plugin_path_str in self.plugin_paths:
            try:
                plugin_path = os.path.abspath(plugin_path_str)
                if os.path.isdir(plugin_path):
                    self._load_plugins_from_directory(plugin_path)
                elif os.path.isfile(plugin_path) and plugin_path.endswith(".py"):
                    self._load_plugin_from_file(plugin_path)
                else:
                    logging.warning(
                        f"Invalid plugin path: {plugin_path_str}. Path is not a directory or a .py file."
                    )
            except Exception as e:
                logging.error(f"Error loading plugins from {plugin_path_str}: {e}")

    def _load_plugins_from_directory(self, plugin_dir: str) -> None:
        """
        Loads plugins from all .py files within a directory.

        :param plugin_dir: Absolute path to the plugin directory.
        :type plugin_dir: str
        :raises ImportError: If there are issues importing plugin modules.
        """
        sys.path.insert(0, plugin_dir)  # Add plugin path to sys.path temporarily
        try:
            for filename in os.listdir(plugin_dir):
                if filename.endswith(".py") and filename != "__init__.py":
                    module_name = filename[:-3]
                    module_path = os.path.join(plugin_dir, filename)
                    self._load_and_register_plugin_module(module_name, module_path)
        finally:
            sys.path.pop(0)  # Remove plugin path from sys.path after loading

    def _load_plugin_from_file(self, plugin_file: str) -> None:
        """
        Loads a plugin from a single .py file.

        :param plugin_file: Absolute path to the plugin file.
        :type plugin_file: str
        :raises ImportError: If there are issues importing the plugin module.
        """
        module_name = os.path.basename(plugin_file[:-3])
        self._load_and_register_plugin_module(module_name, plugin_file)

    def _load_and_register_plugin_module(
        self, module_name: str, module_path: str
    ) -> None:
        """
        Loads a module from a given path and registers plugins found within it.

        :param module_name: Name to use for the module.
        :type module_name: str
        :param module_path: Absolute path to the module file.
        :type module_path: str
        :raises ImportError: If the module cannot be loaded.
        """
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            logging.error(f"Failed to create module spec for {module_path}")
            return
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module  # Register module in sys.modules
        try:
            spec.loader.exec_module(module)  # type: ignore # spec.loader is not None if spec is not None
            self._register_plugins_from_module(module)
        except Exception as e:
            logging.error(f"Error executing module {module_path}: {e}")
            del sys.modules[module_name]  # Clean up sys.modules if module loading fails

    def _register_plugins_from_module(self, module: Any) -> None:
        """
        Registers plugins from a loaded module.

        Inspects the module for classes that are subclasses of :class:`PluginBase`
        (excluding :class:`PluginBase` itself), instantiates them, and adds them
        to the list of managed plugins.

        :param module: The loaded Python module object.
        :type module: Any
        """
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type)
                and issubclass(obj, PluginBase)
                and obj != PluginBase
            ):
                try:
                    plugin_instance = obj()
                    self.plugins.append(plugin_instance)
                    logging.info(
                        f"Loaded plugin: {obj.__name__} from module {module.__name__}"
                    )
                except Exception as e:
                    logging.error(
                        f"Error instantiating plugin {obj.__name__} from module {module.__name__}: {e}"
                    )

    def load_plugin(self, plugin_path: str) -> None:
        """
        Loads a single plugin from a given path and registers it.

        This method allows for loading a plugin on demand, in addition to the
        plugins loaded during initialization.

        :param plugin_path: Path to the plugin file or directory.
        :type plugin_path: str
        """
        try:
            plugin_path = os.path.abspath(plugin_path)
            if os.path.isdir(plugin_path):
                self._load_plugins_from_directory(plugin_path)
            elif os.path.isfile(plugin_path) and plugin_path.endswith(".py"):
                self._load_plugin_from_file(plugin_path)
            else:
                logging.warning(
                    f"Invalid plugin path provided for single plugin load: {plugin_path}"
                )
        except Exception as e:
            logging.error(f"Error loading single plugin from {plugin_path}: {e}")

    def register_plugins(self) -> None:
        """
        Registers hooks for all loaded plugins.

        Iterates through the list of loaded plugins and calls their
        :meth:`PluginBase.register_hooks` method to activate their
        functionality within the application.
        """
        for plugin in self.plugins:
            try:
                plugin.register_hooks()
                logging.info(
                    f"Registered hooks for plugin: {plugin.__class__.__name__}"
                )
            except Exception as e:
                logging.error(
                    f"Error registering hooks for plugin {plugin.__class__.__name__}: {e}"
                )


class DependencyContainer:
    """
    Manages dependency injection, enabling loose coupling between components.

    Supports registration of dependencies as singletons (one instance per container)
    or transients (new instance each time). Dependencies are resolved based on
    type hints in constructor parameters.

    :raises DependencyResolutionError: If a dependency cannot be resolved.
    :raises TypeError: If registration parameters are of incorrect type.
    """

    def __init__(self) -> None:
        """Initializes the DependencyContainer."""
        self.dependencies: Dict[Type, Type] = {}
        """Dictionary mapping interfaces to their implementations."""
        self.singletons: Dict[Type, Any] = {}
        """Dictionary storing singleton instances, keyed by interface type."""

    def register_singleton(self, interface: Type, implementation: Type) -> None:
        """
        Registers a dependency as a singleton.

        A singleton dependency is instantiated only once per :class:`DependencyContainer`
        instance, and the same instance is returned every time it is resolved.

        :param interface: The interface type (typically an abstract base class or Protocol).
        :type interface: Type
        :param implementation: The implementation type (the concrete class).
        :type implementation: Type
        :raises TypeError: If ``interface`` or ``implementation`` is not a type.
        """
        self._validate_registration_types(interface, implementation)
        if interface in self.dependencies and interface not in self.singletons:
            logging.warning(
                f"Overriding transient registration for interface: {interface} with singleton."
            )
        elif (
            interface in self.singletons
            and self.dependencies.get(interface) != implementation
        ):
            logging.warning(
                f"Overriding singleton registration for interface: {interface} with new implementation."
            )

        self.dependencies[interface] = implementation
        self.singletons[interface] = None  # Initialize singleton instance to None

    def register_transient(self, interface: Type, implementation: Type) -> None:
        """
        Registers a dependency as transient.

        A transient dependency is instantiated anew each time it is resolved.

        :param interface: The interface type.
        :type interface: Type
        :param implementation: The implementation type.
        :type implementation: Type
        :raises TypeError: If ``interface`` or ``implementation`` is not a type.
        """
        self._validate_registration_types(interface, implementation)
        if interface in self.dependencies and interface in self.singletons:
            logging.warning(
                f"Overriding singleton registration for interface: {interface} with transient."
            )
        elif (
            interface in self.dependencies
            and self.dependencies.get(interface) != implementation
            and interface not in self.singletons
        ):
            logging.warning(
                f"Overriding transient registration for interface: {interface} with new implementation."
            )

        self.dependencies[interface] = implementation
        if interface in self.singletons:
            del self.singletons[interface]  # Ensure it's not treated as singleton

    def _validate_registration_types(
        self, interface: Type, implementation: Type
    ) -> None:
        """
        Validates that interface and implementation are types during registration.

        :param interface: The interface type.
        :type interface: Type
        :param implementation: The implementation type.
        :type implementation: Type
        :raises TypeError: If ``interface`` or ``implementation`` is not a type.
        """
        if not isinstance(interface, type):
            raise TypeError(f"Interface must be a type, got {type(interface)}")
        if not isinstance(implementation, type):
            raise TypeError(
                f"Implementation must be a type, got {type(implementation)}"
            )

    def resolve(self, interface: Type) -> Any:
        """
        Resolves a dependency and returns an instance of the implementation.

        Looks up the registered implementation for the given interface type.
        If it's a singleton, returns the stored instance (creating it if it doesn't exist).
        If it's transient, creates a new instance.

        :param interface: The interface type to resolve.
        :type interface: Type
        :raises DependencyResolutionError: If no implementation is registered for the interface.
        :return: An instance of the implementation.
        :rtype: Any
        """
        implementation = self.dependencies.get(interface)
        if not implementation:
            raise DependencyResolutionError(
                f"No implementation registered for interface: {interface}"
            )

        if interface in self.singletons:  # Singleton scope
            if self.singletons[interface] is None:
                self.singletons[interface] = self._create_instance(
                    interface, implementation
                )
            return self.singletons[interface]
        else:  # Transient scope
            return self._create_instance(interface, implementation)

    def _create_instance(self, interface: Type, implementation: Type) -> Any:
        """
        Creates an instance of the implementation, resolving constructor dependencies recursively.

        Inspects the constructor of the implementation class for type-hinted parameters.
        For each type-hinted parameter (excluding 'self'), it recursively resolves
        the dependency using the container and passes the resolved instances as
        arguments to the constructor.

        :param interface: The interface type being resolved (for error context).
        :type interface: Type
        :param implementation: The implementation type to instantiate.
        :type implementation: Type
        :raises DependencyResolutionError: If a constructor dependency cannot be resolved.
        :return: An instance of the implementation.
        :rtype: Any
        """
        try:
            constructor_params = implementation.__init__.__signature__.parameters if hasattr(implementation, "__init__") else {}  # type: ignore
            resolved_params = {}
            for param_name, param in constructor_params.items():
                if param_name != "self" and param.annotation != param.empty:
                    try:
                        resolved_params[param_name] = self.resolve(param.annotation)  # type: ignore
                    except DependencyResolutionError as e:
                        raise DependencyResolutionError(
                            f"Failed to resolve dependency '{param_name}' of type '{param.annotation}' for class '{implementation.__name__}' implementing interface '{interface.__name__}'. "
                            + str(e)
                        ) from e
            return implementation(**resolved_params)
        except Exception as e:
            raise DependencyResolutionError(
                f"Error creating instance of '{implementation.__name__}' implementing interface '{interface.__name__}': {e}"
            ) from e


# State Management


class InMemoryPersistenceBackend:
    """
    In-memory persistence backend for :class:`StateManager`.

    Stores state data in a dictionary in memory. This backend is primarily
    intended for testing or non-persistent environments. Data is lost when
    the application terminates. Not suitable for production environments
    requiring data persistence across sessions.
    """

    def __init__(self) -> None:
        """
        Initializes the InMemoryPersistenceBackend with an empty state.

        The internal state dictionary `_state` is initialized to an empty
        dictionary when the backend is created.
        """
        self._state: Dict[str, Any] = {}

    def load_state(self) -> Dict[str, Any]:
        """
        Loads state data from in-memory storage.

        Returns the current in-memory state. If no state has been saved yet,
        it returns an empty dictionary.

        :return: The loaded state data.
        :rtype: Dict[str, Any]
        """
        return self._state

    def save_state(
        self, state_data: Dict[str, Any], task_id: Optional[str] = None
    ) -> None:
        """
        Saves state data to in-memory storage.

        Updates the internal `_state` dictionary with the provided `state_data`.
        Any previous state is overwritten.

        :param state_data: The state data to be saved.
        :type state_data: Dict[str, Any]
        :param task_id: Optional task ID associated with this state saving operation. This can be
                        useful for tracking or auditing state changes. Defaults to None.
        :type task_id: Optional[str]
        """
        self._state = state_data  # Add optional task_id parameter to match interface


class StateManager:
    """
    Orchestration state manager.

    Manages the state of the system, including cluster-wide state and
    node-specific state. State is persisted and retrieved using a
    pluggable persistence backend, allowing for different storage mechanisms.

    The state is divided into two main categories:
        - **Cluster State**: Global state shared across all nodes in the cluster.
        - **Node State**: State specific to individual nodes, identified by a node ID.

    The :class:`StateManager` uses an :class:`IPersistenceBackend` to handle
    the actual loading and saving of state data, abstracting away the
    underlying persistence mechanism.
    """

    def __init__(
        self, persistence_backend: Optional[IPersistenceBackend] = None
    ) -> None:
        """
        Initializes the StateManager.

        Configures the state manager with a persistence backend. If no backend
        is provided, it defaults to :class:`InMemoryPersistenceBackend` for
        non-persistent, in-memory storage.  Loads initial state from the backend
        upon initialization.

        :param persistence_backend: Backend for state persistence. Defaults to
            :class:`InMemoryPersistenceBackend` if None.
        :type persistence_backend: Optional[IPersistenceBackend], optional
        """
        self.persistence_backend: IPersistenceBackend = (
            persistence_backend or InMemoryPersistenceBackend()
        )
        loaded_state = self.persistence_backend.load_state()
        self._cluster_state: Dict[str, Any] = loaded_state.get("cluster_state", {})
        self._node_state: Dict[str, Any] = loaded_state.get("node_state", {})

    def get_cluster_state(self) -> Dict[str, Any]:
        """
        Retrieves the current cluster state.

        Returns a dictionary representing the current cluster-wide state.
        Changes to this dictionary will not affect the managed state; use
        :meth:`update_cluster_state` to modify the state.

        :return: The current cluster state.
        :rtype: Dict[str, Any]
        """
        return self._cluster_state

    def update_cluster_state(self, update_data: Dict[str, Any]) -> None:
        """
        Updates the cluster state and persists it using the backend.

        Merges the provided `update_data` into the current cluster state.
        The updated state is then persisted using the configured
        :class:`IPersistenceBackend`.

        :param update_data: Data to update the cluster state with.
        :type update_data: Dict[str, Any]
        """
        self._cluster_state.update(update_data)
        self._persist_state()

    def get_node_state(self, node_id: str) -> Dict[str, Any]:
        """
        Retrieves the state for a specific node.

        Returns a dictionary representing the state of the specified node.
        If no state exists for the node, an empty dictionary is returned.
        Changes to this dictionary will not affect the managed state; use
        :meth:`update_node_state` to modify node-specific state.

        :param node_id: The ID of the node.
        :type node_id: str
        :return: The state of the specified node.
        :rtype: Dict[str, Any]
        """
        return self._node_state.get(node_id, {})

    def update_node_state(self, node_id: str, update_data: Dict[str, Any]) -> None:
        """
        Updates the state for a specific node and persists it using the backend.

        Merges the provided `update_data` into the state of the node identified
        by `node_id`. If the node does not have an existing state, a new state
        dictionary is created for it. The updated state is then persisted
        using the configured :class:`IPersistenceBackend`.

        :param node_id: The ID of the node.
        :type node_id: str
        :param update_data: Data to update the node state with.
        :type update_data: Dict[str, Any]
        """
        if node_id not in self._node_state:
            self._node_state[node_id] = {}
        self._node_state[node_id].update(update_data)
        self._persist_state()

    def _persist_state(self) -> None:
        """
        Persists the current cluster and node states using the persistence backend.

        Internal method to handle the saving of state data.  Combines the
        cluster state and node states into a single dictionary and delegates
        the saving operation to the configured :class:`IPersistenceBackend`.
        """
        state_to_save: Dict[str, Dict[str, Any]] = {
            "cluster_state": self._cluster_state,
            "node_state": self._node_state,
        }
        self.persistence_backend.save_state(state_to_save)


# Metrics, Logging, Profiling, Tracing and Telemetry


class InMemoryMetricsBackend:
    """
    In-memory metrics backend for :class:`MetricsRegistry`.

    Stores metrics data in a dictionary in memory. This backend is primarily
    intended for development, testing, or non-persistent environments.
    Data is lost when the application terminates. Not suitable for production
    environments requiring persistent metrics storage or export.

    For production use, consider backends like PrometheusMetricsBackend or StatsDMetricsBackend
    (not implemented in this example).
    """

    DEFAULT_HISTOGRAM_BUCKETS: List[float] = [
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        float("inf"),
    ]
    """Default histogram buckets in seconds."""

    def __init__(self) -> None:
        """Initializes the InMemoryMetricsBackend with an empty metrics store."""
        self.metrics: Dict[str, Any] = {}

    def create_counter(
        self,
        name: str,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Creates an in-memory counter metric.

        Args:
            name (str): The name of the counter metric.
            description (Optional[str]): A human-readable description of the metric. Defaults to None.
            labels (Optional[Dict[str, str]]): Optional labels for the metric (not used in this simple in-memory backend). Defaults to None.

        Returns:
            int: The created counter metric object (initialized to 0).

        Raises:
            ValueError: If a metric with the same name already exists.
        """
        if name in self.metrics:
            raise ValueError(f"Metric with name '{name}' already exists.")
        self.metrics[name] = {"type": "counter", "value": 0}
        return self.metrics[name]

    def create_gauge(
        self,
        name: str,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> float:
        """
        Creates an in-memory gauge metric.

        Args:
            name (str): The name of the gauge metric.
            description (Optional[str]): A human-readable description of the metric. Defaults to None.
            labels (Optional[Dict[str, str]]): Optional labels for the metric (not used in this simple in-memory backend). Defaults to None.

        Returns:
            float: The created gauge metric object (initialized to 0.0).

        Raises:
            ValueError: If a metric with the same name already exists.
        """
        if name in self.metrics:
            raise ValueError(f"Metric with name '{name}' already exists.")
        self.metrics[name] = {"type": "gauge", "value": 0.0}
        return self.metrics[name]

    def create_histogram(
        self,
        name: str,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Dict[str, int]:
        """
        Creates an in-memory histogram metric.

        Uses predefined or provided buckets to count value observations.

        Args:
            name (str): The name of the histogram metric.
            description (Optional[str]): A human-readable description of the metric. Defaults to None.
            labels (Optional[Dict[str, str]]): Optional labels for the metric (not used in this simple in-memory backend). Defaults to None.
            buckets (Optional[List[float]]): Custom bucket boundaries. Defaults to :attr:`DEFAULT_HISTOGRAM_BUCKETS`.

        Returns:
            Dict[str, int]: The created histogram metric object (dictionary representing buckets and counts).

        Raises:
            ValueError: If a metric with the same name already exists.
        """
        if name in self.metrics:
            raise ValueError(f"Metric with name '{name}' already exists.")
        buckets_to_use = (
            buckets if buckets is not None else self.DEFAULT_HISTOGRAM_BUCKETS
        )
        histogram_data: Dict[str, int] = {str(bucket): 0 for bucket in buckets_to_use}
        self.metrics[name] = {
            "type": "histogram",
            "buckets": histogram_data,
            "bucket_limits": buckets_to_use,
        }
        return self.metrics[name]

    def increment_counter(
        self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increments an in-memory counter metric.

        Args:
            name (str): The name of the counter metric.
            value (int): The value to increment by. Defaults to 1.
            labels (Optional[Dict[str, str]]): Optional labels (not used in this simple in-memory backend). Defaults to None.

        Raises:
            ValueError: If the metric is not a counter or does not exist.
        """
        if name not in self.metrics:
            raise ValueError(f"Metric with name '{name}' does not exist.")
        metric_data = self.metrics[name]
        if metric_data["type"] != "counter":
            raise ValueError(f"Metric '{name}' is not a counter.")
        metric_data["value"] += value

    def set_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Sets the value of an in-memory gauge metric.

        Args:
            name (str): The name of the gauge metric.
            value (float): The value to set the gauge to.
            labels (Optional[Dict[str, str]]): Optional labels (not used in this simple in-memory backend). Defaults to None.

        Raises:
            ValueError: If the metric is not a gauge or does not exist.
        """
        if name not in self.metrics:
            raise ValueError(f"Metric with name '{name}' does not exist.")
        metric_data = self.metrics[name]
        if metric_data["type"] != "gauge":
            raise ValueError(f"Metric '{name}' is not a gauge.")
        metric_data["value"] = value

    def observe_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Observes a value in an in-memory histogram metric.

        Finds the correct bucket for the value and increments its count.

        Args:
            name (str): The name of the histogram metric.
            value (float): The value to observe.
            labels (Optional[Dict[str, str]]): Optional labels (not used in this simple in-memory backend). Defaults to None.

        Raises:
            ValueError: If the metric is not a histogram or does not exist.
        """
        if name not in self.metrics:
            raise ValueError(f"Metric with name '{name}' does not exist.")
        metric_data = self.metrics[name]
        if metric_data["type"] != "histogram":
            raise ValueError(f"Metric '{name}' is not a histogram.")

        buckets = metric_data["bucket_limits"]
        histogram_counts = metric_data["buckets"]
        bucket_found = False
        for bucket_limit in buckets:
            if value <= bucket_limit:
                histogram_counts[str(bucket_limit)] += 1
                bucket_found = True
                break
        if (
            not bucket_found
        ):  # Should not happen if buckets are correctly configured with +inf
            histogram_counts[
                str(buckets[-1])
            ] += 1  # Fallback to the last bucket (+inf)


class MetricsRegistry:
    """
    Unified metrics registry.

    Collects and manages metrics of different types (:class:`MetricTypes`).
    Provides a consistent interface to record metrics, regardless of the
    underlying metrics backend.

    Metrics can be exported to various backends (e.g., in-memory, Prometheus, StatsD)
    via a configured :class:`IMetricsBackend`. This class acts as a facade,
    simplifying metric operations for the application code.
    """

    def __init__(self, metrics_backend: Optional[IMetricsBackend] = None) -> None:
        """
        Initializes the MetricsRegistry.

        Args:
            metrics_backend (Optional[IMetricsBackend]): Backend for metrics storage and export.
                                                        Defaults to :class:`InMemoryMetricsBackend` if None.
        """
        self.metrics_backend: IMetricsBackend = (
            metrics_backend or InMemoryMetricsBackend()
        )
        """Backend for storing and exporting metrics data."""
        self.metrics: Dict[str, Any] = {}  # Metric name to metric object mapping

    def create_metric(
        self,
        name: str,
        metric_type: MetricTypes,
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[List[float]] = None,
    ) -> Any:
        """
        Creates a new metric in the registry.

        This method acts as a factory for creating metrics, delegating the actual
        creation to the configured :attr:`metrics_backend`.

        Args:
            name (str): The name of the metric. Must be unique within the registry.
            metric_type (:class:`MetricTypes`): The type of the metric (COUNTER, GAUGE, HISTOGRAM).
            description (Optional[str]): A human-readable description of the metric. Defaults to None.
            labels (Optional[Dict[str, str]]): Optional labels (dimensions) to attach to the metric. Defaults to None.
            buckets (Optional[List[float]]): For HISTOGRAM metrics, custom bucket boundaries. Defaults to None, using backend defaults.

        Returns:
            Any: The created metric object. The type depends on the backend and metric type.

        Raises:
            ValueError: If a metric with the same name already exists.
            ValueError: If an unsupported metric type is provided.
        """
        if name in self.metrics:
            raise ValueError(f"Metric with name '{name}' already exists.")

        if metric_type == MetricTypes.COUNTER:
            metric = self.metrics_backend.create_counter(name, description, labels)
        elif metric_type == MetricTypes.GAUGE:
            metric = self.metrics_backend.create_gauge(name, description, labels)
        elif metric_type == MetricTypes.HISTOGRAM:
            metric = self.metrics_backend.create_histogram(
                name, description, labels, buckets
            )
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")
        self.metrics[name] = metric
        return metric  # Return the created metric object

    def increment_counter(
        self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increments a counter metric.

        Delegates the increment operation to the configured :attr:`metrics_backend`.

        Args:
            name (str): The name of the counter metric.
            value (int): The value to increment by. Defaults to 1.
            labels (Optional[Dict[str, str]]): Optional labels to apply when incrementing. Must match the labels defined at creation. Defaults to None.

        Raises:
            ValueError: If the metric is not a counter or does not exist.
        """
        self.metrics_backend.increment_counter(name, value, labels)

    def set_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Sets the value of a gauge metric.

        Delegates the set operation to the configured :attr:`metrics_backend`.

        Args:
            name (str): The name of the gauge metric.
            value (float): The value to set the gauge to.
            labels (Optional[Dict[str, str]]): Optional labels to apply when setting the gauge. Must match the labels defined at creation. Defaults to None.

        Raises:
            ValueError: If the metric is not a gauge or does not exist.
        """
        self.metrics_backend.set_gauge(name, value, labels)

    def observe_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Observes a value in a histogram metric.

        Delegates the observe operation to the configured :attr:`metrics_backend`.

        Args:
            name (str): The name of the histogram metric.
            value (float): The value to observe.
            labels (Optional[Dict[str, str]]): Optional labels to apply when observing. Must match the labels defined at creation. Defaults to None.

        Raises:
            ValueError: If the metric is not a histogram or does not exist.
        """
        self.metrics_backend.observe_histogram(name, value, labels)


# Resources and Telemetry and Profiling


class ResourceMonitor(IResourceMonitor):  # Inherit from interface
    """
    Real-time system resource monitor providing detailed insights into CPU, memory,
    disk, network, and GPU utilization.

    Establishes baselines for resource usage and monitors deviations, facilitating
    anomaly detection and performance analysis. Designed to be modular and extensible,
    allowing for customization of monitored resources and anomaly detection strategies.

    .. note::
       Anomaly detection is a future feature and currently provides basic baseline comparison.
    """

    def __init__(
        self,
        monitoring_interval: float = 1.0,
        disk_path: str = "/",
        anomaly_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Initializes the ResourceMonitor with configurable monitoring interval and disk path.

        :param monitoring_interval: Interval in seconds between resource monitoring cycles. Must be greater than 0. Defaults to 1.0 second.
        :type monitoring_interval: float
        :param disk_path: Filesystem path to monitor disk usage for. Defaults to root ("/").
        :type disk_path: str
        :param anomaly_thresholds: Dictionary of thresholds for anomaly detection. Keys are resource names (e.g., 'cpu_percent'), values are maximum allowed deviation from baseline (percentage). Defaults to None (no anomaly detection).
        :type anomaly_thresholds: Optional[Dict[str, float]]
        :raises ValueError: If ``monitoring_interval`` is not greater than 0.
        """
        if monitoring_interval <= 0:
            raise ValueError("monitoring_interval must be greater than 0")
        self.monitoring_interval = monitoring_interval
        self.disk_path = disk_path
        self.baselines: Dict[str, float] = {}
        self._is_baseline_established = False
        self.anomaly_thresholds = anomaly_thresholds or {}

    def start_monitoring(self, interval: int = 5) -> None:
        """
        Starts resource monitoring with a specified interval and establishes baselines.

        :param interval: Interval in seconds between monitoring cycles. Defaults to 5 seconds.
        :type interval: int
        """
        if interval <= 0:
            raise ValueError("Monitoring interval must be greater than 0")
        self.monitoring_interval = float(
            interval
        )  # Ensure monitoring_interval is updated
        logging.info(f"Resource monitoring started with interval: {interval} seconds.")
        # Baselines are established upon the first call to get_usage or monitor_resources
        # or explicitly calling establish_baselines if needed immediately after start_monitoring.

    async def establish_baselines(self) -> None:
        """
        Asynchronously measures and establishes initial system resource baselines.

        This method should be called once at application startup to capture normal
        resource utilization. It records CPU, memory, disk, network, and GPU usage
        as baseline values for future monitoring and anomaly detection.
        """
        self.baselines = await self._get_current_resources()
        self._is_baseline_established = True
        logging.info("Resource baselines established.")

    async def _get_current_resources(self) -> Dict[str, float]:
        """
        Asynchronously retrieves current system resource metrics.

        This is a utility method to collect real-time resource usage data for CPU,
        memory, disk, network I/O, and GPU. It encapsulates the calls to system
        monitoring libraries (e.g., `psutil`, `GPUtil`).

        :return: Dictionary of current resource metrics with keys like 'cpu_percent', 'memory_percent', etc.
        :rtype: Dict[str, float]
        """
        return {
            "cpu_percent": self._get_cpu_percent(),
            "memory_percent": self._get_memory_percent(),
            "disk_usage_percent": self._get_disk_usage_percent(),
            "network_bytes_sent": self._get_network_bytes_sent(),
            "network_bytes_recv": self._get_network_bytes_recv(),
            "gpu_mem_percent": self._get_gpu_utilization(),
        }

    def _get_cpu_percent(self) -> float:
        """
        Retrieves the current CPU utilization percentage.

        :return: CPU utilization percentage.
        :rtype: float
        """
        return psutil.cpu_percent()

    def _get_memory_percent(self) -> float:
        """
        Retrieves the current memory utilization percentage.

        :return: Memory utilization percentage.
        :rtype: float
        """
        return psutil.virtual_memory().percent

    def _get_disk_usage_percent(self) -> float:
        """
        Retrieves the current disk usage percentage for the configured path.

        :return: Disk usage percentage.
        :rtype: float
        """
        return psutil.disk_usage(self.disk_path).percent

    def _get_network_bytes_sent(self) -> float:
        """
        Retrieves the current number of bytes sent over the network since boot.

        :return: Network bytes sent.
        :rtype: float
        """
        return psutil.net_io_counters().bytes_sent

    def _get_network_bytes_recv(self) -> float:
        """
        Retrieves the current number of bytes received over the network since boot.

        :return: Network bytes received.
        :rtype: float
        """
        return psutil.net_io_counters().bytes_recv

    def _get_gpu_utilization(self) -> float:
        """
        Retrieves GPU memory utilization percentage using GPUtil.

        Returns 0.0 if GPUtil is not available or if no GPUs are found.

        :return: GPU memory utilization percentage, or 0.0 if GPU monitoring is not available.
        :rtype: float
        """
        try:
            import GPUtil  # type: ignore  # Import GPUtil only when needed

            gpus = GPUtil.getGPUs()
            if gpus:
                return (
                    gpus[0].memoryUtil * 100
                )  # Uses the first GPU, consider more sophisticated selection if needed
            return 0.0
        except ImportError:
            logging.warning("GPUtil library not found. GPU monitoring disabled.")
            return 0.0
        except Exception as e:
            logging.error(f"Error getting GPU utilization: {e}")
            return 0.0

    async def monitor_resources(self) -> Dict[str, float]:
        """
        Asynchronously monitors current system resources and compares them against established baselines.

        If baselines have not been established yet, it establishes them before monitoring.
        Optionally detects anomalies based on configured thresholds.

        :return: Dictionary of current resource metrics.
        :rtype: Dict[str, float]
        """
        if not self._is_baseline_established:
            await self.establish_baselines()  # Ensure baselines are established

        current_resources = await self._get_current_resources()
        if self.anomaly_thresholds and self._is_baseline_established:
            self._check_anomalies(
                current_resources
            )  # Basic anomaly check, can be extended

        return current_resources

    def _check_anomalies(self, current_resources: Dict[str, float]) -> None:
        """
        Checks for resource anomalies by comparing current usage to baselines.

        Logs warnings if current resource usage deviates from the baseline by more than
        the configured threshold for any resource specified in ``anomaly_thresholds``.

        :param current_resources: Dictionary of current resource metrics.
        :type current_resources: Dict[str, float]
        """
        for resource_name, threshold in self.anomaly_thresholds.items():
            if resource_name in current_resources and resource_name in self.baselines:
                deviation = abs(
                    current_resources[resource_name] - self.baselines[resource_name]
                )
                if deviation > threshold:
                    logging.warning(
                        f"Resource anomaly detected: {resource_name} deviation from baseline is {deviation:.2f}%, exceeding threshold {threshold:.2f}%."
                    )

    async def get_usage(self) -> Dict[str, float]:  # Keep async implementation
        """
        Asynchronously retrieves the current resource usage.

        Implements the get_usage method from the IResourceMonitor interface.

        :return: A dictionary containing current resource usage metrics.
        :rtype: Dict[str, float]
        """
        if not self._is_baseline_established:
            await self.establish_baselines()  # Establish baselines before getting usage if not already done
        return await self._get_current_resources()


class InMemoryTelemetryBackend(ITelemetryBackend):
    """
    In-memory telemetry backend for :class:`TelemetryCollector`, primarily for development and testing.

    Stores telemetry span data in memory, which is transient and lost when the application terminates.
    Not suitable for production environments requiring persistent telemetry data or export to external systems.
    """

    def __init__(self) -> None:
        """Initializes the InMemoryTelemetryBackend with an empty dictionary to store spans."""
        self.spans: Dict[str, "TelemetrySpan"] = {}
        self.metrics: Dict[str, float] = {}  # Initialize metrics storage
        self.events: List[Dict[str, Any]] = []  # Initialize events storage
        self.logging = logging.getLogger(__name__)

    def start_span(
        self,
        task_id: str,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "TelemetrySpan":
        """
        Starts an in-memory telemetry span.

        :param task_id: Unique identifier for the task.
        :type task_id: str
        :param operation_name: A descriptive name for the operation.
        :type operation_name: str
        :param attributes: Optional attributes to attach to the span. Defaults to None.
        :type attributes: Optional[Dict[str, Any]]
        :return: The newly created TelemetrySpan object.
        :rtype: TelemetrySpan
        """
        span = TelemetrySpan(task_id, backend=self, attributes=attributes)
        self.spans[task_id] = span
        self.logging.debug(
            f"Starting telemetry span for task: {task_id} with attributes: {attributes}"
        )
        return span

    def end_span(
        self, span: "TelemetrySpan", attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Ends an in-memory telemetry span.

        :param span: The TelemetrySpan object to end.
        :type span: TelemetrySpan
        :param attributes: Optional attributes to add when ending the span. Defaults to None.
        :type attributes: Optional[Dict[str, Any]]
        """
        if span.task_id in self.spans:
            del self.spans[span.task_id]
            self.logging.debug(
                f"Telemetry span ended for task: {span.task_id} with final attributes: {attributes}"
            )
        else:
            self.logging.warning(
                f"Attempted to end non-existent telemetry span for task: {span.task_id}"
            )

    def record_metric(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Records a telemetry metric in memory.

        :param name: Name of the metric.
        :type name: str
        :param value: Numeric value of the metric.
        :type value: float
        :param tags: Optional dictionary of key-value tags for metric categorization. Defaults to None.
        :type tags: Optional[Dict[str, str]]
        """
        self.metrics[name] = value  # Simply store the latest value
        tags_str = f" with tags: {tags}" if tags else ""
        self.logging.debug(f"Metric recorded: {name}={value}{tags_str}")

    def capture_event(
        self, event_name: str, properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Captures a system event in memory.

        :param event_name: Name of the event.
        :type event_name: str
        :param properties: Optional dictionary of event properties. Defaults to None.
        :type properties: Optional[Dict[str, Any]]
        """
        event_data = {"event_name": event_name, "properties": properties}
        self.events.append(event_data)
        properties_str = f" with properties: {properties}" if properties else ""
        self.logging.debug(f"Event captured: {event_name}{properties_str}")


class TelemetryCollector(ITelemetryCollector):
    """
    Collects and manages telemetry data using a configurable backend.

    Provides methods to start and end telemetry spans, allowing for the measurement
    of task durations and the collection of contextual attributes. Supports various
    telemetry backends through the :class:`ITelemetryBackend` interface, enabling
    export to different monitoring and analysis systems.
    """

    def __init__(self, telemetry_backend: Optional[ITelemetryBackend] = None) -> None:
        """
        Initializes the TelemetryCollector with a specified telemetry backend.

        Defaults to :class:`InMemoryTelemetryBackend` if no backend is provided.

        :param telemetry_backend: Backend for telemetry data export and storage. Defaults to InMemoryTelemetryBackend.
        :type telemetry_backend: Optional[ITelemetryBackend]
        """
        self.telemetry_backend = telemetry_backend or InMemoryTelemetryBackend()
        self.spans: Dict[str, "TelemetrySpan"] = (
            {}
        )  # Maps task IDs to active TelemetrySpan instances

    def start_span(
        self,
        task_id: str,
        operation_name: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> "TelemetrySpan":
        """
        Starts an in-memory telemetry span.

        :param task_id: Unique identifier for the task.
        :type task_id: str
        :param operation_name: A descriptive name for the operation.
        :type operation_name: str
        :param attributes: Optional attributes to attach to the span. Defaults to None.
        :type attributes: Optional[Dict[str, Any]]
        :return: The newly created TelemetrySpan object.
        :rtype: TelemetrySpan
        """
        span = TelemetrySpan(task_id, backend=self, attributes=attributes)
        self.spans[task_id] = span
        logging.debug(
            f"Starting telemetry span for task: {task_id} with attributes: {attributes}"
        )
        return span

    def end_span(
        self, span: "TelemetrySpan", attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Ends an in-memory telemetry span.

        :param span: The TelemetrySpan object to end.
        :type span: TelemetrySpan
        :param attributes: Optional attributes to add when ending the span. Defaults to None.
        :type attributes: Optional[Dict[str, Any]]
        """
        if span.task_id in self.spans:
            del self.spans[span.task_id]
            logging.debug(
                f"Telemetry span ended for task: {span.task_id} with final attributes: {attributes}"
            )
        else:
            logging.warning(
                f"Attempted to end non-existent telemetry span for task: {span.task_id}"
            )

    def record_metric(
        self, name: str, value: float, tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Records a telemetry metric using the configured backend.

        :param name: Name of the metric.
        :type name: str
        :param value: Numeric value of the metric.
        :type value: float
        :param tags: Optional dictionary of key-value tags for metric categorization. Defaults to None.
        :type tags: Optional[Dict[str, str]]
        """
        self.telemetry_backend.record_metric(name, value, tags)

    def capture_event(
        self, event_name: str, properties: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Captures a system event using the configured backend.

        :param event_name: Name of the event.
        :type event_name: str
        :param properties: Optional dictionary of properties associated with the event. Defaults to None.
        :type properties: Optional[Dict[str, Any]]
        """
        self.telemetry_backend.capture_event(event_name, properties)


class TelemetrySpan:
    """
    Represents a telemetry span for a specific task, measuring its duration and collecting attributes.

    Designed to be used as a context manager to automatically handle span start and end times.
    Provides a mechanism to attach attributes to spans for richer telemetry data.
    """

    def __init__(
        self,
        task_id: str,
        backend: Optional[InMemoryTelemetryBackend | ITelemetryBackend] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes a TelemetrySpan.

        :param task_id: Unique identifier for the task associated with this span.
        :type task_id: str
        :param backend: The telemetry backend to use for reporting the span end. Defaults to None, expecting a backend from TelemetryCollector.
        :type backend: Optional[InMemoryTelemetryBackend]
        :param attributes: Initial attributes to attach to the span. Defaults to None.
        :type attributes: Optional[Dict[str, Any]]
        """
        self.task_id = task_id
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.backend = backend
        self.attributes: Dict[str, Any] = attributes or {}

    def __enter__(self) -> "TelemetrySpan":
        """
        Enters the telemetry span context.

        Records the start time of the span.

        :return: The TelemetrySpan instance itself, allowing for context management.
        :rtype: TelemetrySpan
        """
        self.start_time = time.time()
        logging.debug(
            f"Entering telemetry span for task: {self.task_id} with attributes: {self.attributes}"
        )
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Exits the telemetry span context.

        Records the end time, calculates the duration, logs the span details, and informs
        the telemetry backend to finalize the span. Allows adding attributes at span end.

        :param exc_type: Exception type if an exception occurred within the context.
        :type exc_type: Optional[Type[BaseException]]
        :param exc_val: Exception value if an exception occurred.
        :type exc_val: Optional[BaseException]
        :param exc_tb: Exception traceback if an exception occurred.
        :type exc_tb: Optional[Any]
        :param attributes: Optional attributes to add when ending the span, merged with initial attributes. Defaults to None.
        :type attributes: Optional[Dict[str, Any]]
        """
        self.end_time = time.time()
        duration = (self.end_time or 0) - (self.start_time or 0)
        if attributes:
            self.attributes.update(
                attributes
            )  # Merge end attributes with existing ones
        logging.debug(
            f"Telemetry span ended for task: {self.task_id}, duration: {duration:.4f}s, final attributes: {self.attributes}"
        )
        if self.backend:
            self.backend.end_span(self, attributes=self.attributes)


class Profiler(IProfiler):  # Inherit from interface
    """
    Performance profiling tool using cProfile to analyze task execution.

    Profiles a given task function and saves a detailed profiling report to a file.
    Allows customization of the output path for profile reports and sorting of profiling statistics.
    """

    def __init__(self, output_path: str = "profile_reports") -> None:
        """
        Initializes the Profiler with a specified output path for reports.

        :param output_path: Directory path where profile reports will be saved. Defaults to "profile_reports".
        :type output_path: str
        """
        self.output_path = output_path
        os.makedirs(
            self.output_path, exist_ok=True
        )  # Ensure the output directory exists

    def start_profiling(self, context: Dict[str, Any]) -> cProfile.Profile:
        """
        Starts profiling using cProfile.

        :param context: Dictionary containing context information for profiling, e.g., task_id, sort_by.
        :type context: Dict[str, Any]
        :return: cProfile.Profile object that has been started.
        :rtype: cProfile.Profile
        """
        profile = cProfile.Profile()
        profile.enable()
        logging.debug(f"Profiling started with context: {context}")
        return profile

    def stop_profiling(
        self, profiler: cProfile.Profile, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Stops profiling, saves the report, and returns profiling results.

        :param profiler: The cProfile.Profile object to stop and finalize.
        :type profiler: cProfile.Profile
        :param context: Dictionary containing context information for profiling, e.g., task_id, sort_by.
        :type context: Dict[str, Any]
        :return: Dictionary containing profiling results or metadata. Currently returns an empty dict.
        :rtype: Dict[str, Any]
        """
        profiler.disable()
        task_id = context.get("task_id", "default_task")
        sort_by = context.get("sort_by", "cumulative")
        report_file = os.path.join(
            self.output_path, f"profile_{task_id}_{time.strftime('%Y%m%d-%H%M%S')}.txt"
        )
        try:
            with open(report_file, "w") as f:
                sys.stdout = f  # Redirect stdout to the report file
                profiler.print_stats(sort=sort_by)
            logging.info(
                f"Profiling report saved to: {report_file} sorted by: {sort_by}"
            )
        except ValueError as e:
            logging.error(f"Invalid sort_by option '{sort_by}' for profiling: {e}")
        finally:
            sys.stdout = sys.__stdout__  # Restore standard output
        return {}

    def profile_task(
        self, task: Callable[..., Any], task_id: str, sort_by: str = "cumulative"
    ) -> Any:
        """
        Profiles the execution of a given task function and saves a report.

        :param task: The function to be profiled.
        :type task: Callable[..., Any]
        :param task_id: A unique identifier for the task being profiled.
        :type task_id: str
        :param sort_by: The sorting criterion for the profiling report.
        :type sort_by: str
        :raises ValueError: If ``sort_by`` is not a valid sorting option for ``cProfile.Stats``.
        :return: The result of executing the profiled task function.
        :rtype: Any
        """
        context = {"task_id": task_id, "sort_by": sort_by}
        profiler = self.start_profiling(context)
        result = task()  # Execute the task
        self.stop_profiling(profiler, context)
        return result  # Return the result of the profiled task


# Error Handling and Resilience

#### Tracing Components


class InMemoryTracingBackend:
    """
    In-memory tracing backend for :class:`Tracing`, using Python's :py:mod:`logging` module.

    This backend logs trace messages using the standard logging library. It is
    primarily intended for development, testing, or simple applications where
    persistent or external tracing is not required. For production environments,
    consider using backends that export traces to dedicated tracing systems
    like Jaeger or Zipkin.
    """

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        logging_level: int = logging.DEBUG,
    ) -> None:
        """
        Initializes the InMemoryTracingBackend.

        :param logger: Optional logging instance to use for tracing. If None, a default logger is created.
        :type logger: Optional[logging.Logger]
        :param logging_level: Logging level for trace messages. Defaults to ``logging.DEBUG``.
        :type logging_level: int
        """
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.logging_level = logging_level

    def trace(
        self, task_id: str, message: str, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Logs a trace message using the configured :py:mod:`logging` logging.

        Formats the trace message to include the task ID and then logs it
        at the configured logging level.

        :param task_id: The ID of the task being traced.
        :type task_id: str
        :param message: The trace message to log.
        :type message: str
        :param attributes: Optional attributes to attach to the trace message, providing
                           additional contextual information. Defaults to None.
        :type attributes: Optional[Dict[str, Any]]
        :rtype: None
        """
        log_message = f"[Trace] Task: {task_id} - {message}"
        self.logger.log(self.logging_level, log_message)


class Tracing:
    """
    Tracing tool for instrumenting task execution with detailed logs.

    Provides a consistent interface for logging trace messages throughout
    an application. It is designed to be backend-agnostic, allowing for
    different tracing backends to be plugged in via the :class:`ITracingBackend`
    interface. This enables flexibility in choosing where and how trace data
    is stored and processed (e.g., in-memory, file, or external tracing systems).
    """

    def __init__(self, tracing_backend: Optional[ITracingBackend] = None) -> None:
        """
        Initializes the Tracing tool.

        Configures the tracing tool with a backend that handles the actual
        logging or export of trace messages. If no backend is provided, it
        defaults to :class:`InMemoryTracingBackend`.

        :param tracing_backend: Backend for tracing data export, conforming to :class:`ITracingBackend`.
        :type tracing_backend: Optional[ITracingBackend], optional
        """
        self.tracing_backend: ITracingBackend = (
            tracing_backend or InMemoryTracingBackend()
        )

    def trace(
        self,
        task_id: str,
        message: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Logs a trace message with task context using the configured tracing backend.

        Delegates the actual logging to the configured :attr:`tracing_backend`.

        :param task_id: The ID of the task being traced. This ID should uniquely
                        identify the task or operation being monitored.
        :type task_id: str
        :param message: The trace message to log. This should be a string that
                        provides useful information about the execution flow.
        :type message: str
        :param attributes: Optional attributes to attach to the trace message, providing
                           additional contextual information. Defaults to None.
        :type attributes: Optional[Dict[str, Any]]
        """
        self.tracing_backend.trace(task_id, message, attributes=attributes)


#### Validation Components


class ValidationEngine(ValidationEngineInterface):
    """
    Engine for validating outputs against predefined criteria using pluggable validator functions.
    This engine is enhanced to be context-aware, allowing validator functions to utilize
    contextual information during the validation process.

    This engine allows for registering multiple validator functions, each identified
    by a unique name. When validating an output, it can apply a subset or all of
    the registered validators. This modular design enables flexible and extensible
    validation logic within applications.
    """

    def __init__(
        self,
        default_validation_result: bool = True,
        logging_level: int = logging.WARNING,
    ) -> None:
        """
        Initializes the ValidationEngine.

        :param default_validation_result: The default result to return if no validators are registered.
                                         Defaults to ``True`` (valid if no validators).
        :type default_validation_result: bool
        :param logging_level: Logging level for validation warnings and errors. Defaults to ``logging.WARNING``.
        :type logging_level: int
        """
        self.validators: Dict[str, Callable[[Any, str, Dict[str, Any]], bool]] = {}
        self.default_validation_result = default_validation_result
        self.logging_level = logging_level
        self.logging = logging.getLogger(__name__)

    def register_validator(
        self,
        validator_name: str,
        validator_func: Callable[[Any, str, Dict[str, Any]], bool],
    ) -> None:
        """
        Registers a validator function with the engine.

        Each validator is registered with a unique name, which can be used to
        selectively apply validators during the validation process.
        The validator function is enhanced to accept a context dictionary.

        :param validator_name: The name of the validator. Must be unique within the engine.
        :type validator_name: str
        :param validator_func: The validator function. It should accept the output to be
                             validated, the task ID, and a context dictionary as arguments,
                             and return ``True`` if the output is considered valid, ``False`` otherwise.
        :type validator_func: Callable[[Any, str, Dict[str, Any]], bool]
        :raises TypeError: if ``validator_name`` is not a string or if ``validator_func`` is not callable.
        """
        if not isinstance(validator_name, str):
            raise TypeError(
                f"validator_name must be a string, got {type(validator_name)}"
            )
        if not callable(validator_func):
            raise TypeError(
                f"validator_func must be callable, got {type(validator_func)}"
            )
        self.validators[validator_name] = validator_func

    def validate(
        self,
        output: Any,
        task_id: str,
        context: Dict[str, Any],
        validator_names: Optional[List[str]] = None,
    ) -> bool:
        """
        Validates the output against specified validators, now with context awareness.

        Applies the validators specified by ``validator_names`` to the given
        ``output``, utilizing the provided ``context``. If ``validator_names`` is ``None``,
        all registered validators are applied. If no validators are found for the given names
        or if no validators are registered at all, the behavior is determined by
        :attr:`default_validation_result`.

        :param output: The output to be validated. This can be of any type, depending
                       on what the registered validators are designed to handle.
        :type output: Any
        :param task_id: The ID of the task that produced the output. This can be used
                        by validators to provide context-specific validation.
        :type task_id: str
        :param context: Dictionary containing contextual information relevant to the validation.
        :type context: Dict[str, Any]
        :param validator_names: List of validator names to apply. If ``None``, all
                                registered validators are applied. If an empty list is provided, no validators are run.
        :type validator_names: Optional[List[str]], optional
        :return: ``True`` if all specified validations pass (or if no validators are specified and
                 :attr:`default_validation_result` is ``True``), ``False`` otherwise.
        :rtype: bool
        """
        validators_to_run: List[Callable[[Any, str, Dict[str, Any]], bool]] = []
        if validator_names is None:
            validators_to_run = list(
                self.validators.values()
            )  # Run all if names is None
        elif (
            validator_names
        ):  # Only process if validator_names is not None and not empty
            validators_to_run = [
                self.validators[name]
                for name in validator_names
                if name in self.validators
            ]

        if not validators_to_run:
            self.logging.log(
                self.logging_level,
                f"No validators found for task: {task_id}. Default validation result: {self.default_validation_result}",
            )
            return (
                self.default_validation_result
            )  # Return default if no validators to run

        for validator_func in validators_to_run:
            try:
                if not validator_func(output, task_id, context):
                    self.logging.log(
                        self.logging_level,
                        f"Validation failed for task: {task_id} with validator: {validator_func.__name__}",
                    )
                    return False
            except Exception as e:
                self.logging.error(
                    f"Validator {validator_func.__name__} for task {task_id} raised an exception: {e}",
                    exc_info=True,  # Include exception info in log
                )
                return False  # Validation failed due to error

        self.logging.debug(f"Output validated successfully for task: {task_id}.")
        return True


#### Fallback Handling Components


class FallbackHandler(FallbackHandlerInterface):
    """
    Handles fallback strategies when primary processing fails.

    Provides a mechanism to register and execute different fallback strategies
    based on task IDs or strategy names. This allows for graceful degradation
    and error recovery in applications by providing alternative processing paths
    when the primary path encounters failures.
    """

    def __init__(
        self,
        default_strategy_name: str = "default",
        logging_level: int = logging.WARNING,
    ) -> None:
        """
        Initializes the FallbackHandler.

        :param default_strategy_name: The name of the default fallback strategy to use
                                      when no strategy name is explicitly provided to :meth:`handle_fallback`.
                                      Defaults to "default".
        :type default_strategy_name: str
        :param logging_level: Logging level for fallback handler operations and errors. Defaults to ``logging.WARNING``.
        :type logging_level: int
        """
        self.fallback_strategies: Dict[str, Callable[[str], Any]] = {}
        self.default_strategy_name = default_strategy_name
        self.logging_level = logging_level
        self.logging = logging.getLogger(__name__)

    def register_fallback_strategy(
        self, strategy_name: str, fallback_func: Callable[[str], Any]
    ) -> None:
        """
        Registers a fallback strategy function.

        Associates a given fallback function with a strategy name. This name can
        later be used to invoke this specific strategy via :meth:`handle_fallback`.

        :param strategy_name: The name of the fallback strategy. Must be unique within the handler.
        :type strategy_name: str
        :param fallback_func: The fallback function. It should accept the task ID as
                             input and return a fallback result. It can be synchronous or asynchronous.
        :type fallback_func: Callable[[str], Any]
        :raises TypeError: if ``strategy_name`` is not a string or if ``fallback_func`` is not callable.
        """
        if not isinstance(strategy_name, str):
            raise TypeError(
                f"strategy_name must be a string, got {type(strategy_name)}"
            )
        if not callable(fallback_func):
            raise TypeError(
                f"fallback_func must be callable, got {type(fallback_func)}"
            )
        self.fallback_strategies[strategy_name] = fallback_func

    async def handle_fallback(
        self,
        task_id: str,
        context: Dict[str, Any],
        strategy_name: Optional[str] = None,  # Add interface-required parameter
    ) -> Any:
        """Interface-compliant fallback handling with strategy selection"""
        effective_strategy = (
            strategy_name or context.get("strategy_name") or self.default_strategy_name
        )
        return await self._execute_strategy(task_id, effective_strategy)

    async def _execute_strategy(self, task_id: str, strategy_name: str) -> Any:
        """
        Internal method to execute a fallback strategy.

        :param task_id: The ID of the task that failed and requires a fallback.
        :type task_id: str
        :param strategy_name: The name of the fallback strategy to use.
        :type strategy_name: str
        :return: The result of the fallback strategy execution.
        :rtype: Any
        """
        strategy_func = self.fallback_strategies.get(strategy_name)
        if not strategy_func:
            self.logging.log(
                self.logging_level,
                f"No fallback strategy found for task: {task_id} and strategy name: {strategy_name}. Returning default error.",
            )
            return {"status": "error", "message": "No fallback strategy available."}

        try:
            self.logging.info(
                f"Fallback handler engaged for task: {task_id} using strategy: {strategy_name}"
            )
            fallback_result = (
                await strategy_func(task_id)
                if asyncio.iscoroutinefunction(strategy_func)
                else strategy_func(task_id)
            )
            return fallback_result
        except Exception as e:
            self.logging.error(
                f"Fallback strategy '{strategy_name}' for task {task_id} failed with error: {e}",
                exc_info=True,  # Include exception info in log
            )
            return {"status": "fallback_error", "error_message": str(e)}

    def execute_fallback(
        self, task_id: str, strategy_name: Optional[str] = None
    ) -> Any:
        """Interface-compliant fallback execution"""
        return self.handle_fallback(task_id, {}, strategy_name)


#### Error Handling Components


class ErrorHandler(IErrorHandler):
    """
    Centralized error handler for managing errors across different application modules.

    Provides a structured way to handle errors, log them, and invoke fallback
    strategies using a :class:`FallbackHandler`. It maintains an error log for
    tracking occurred errors and supports configurable logging levels.

    This class now correctly implements the IErrorHandler interface, ensuring
    type compatibility and adherence to the defined contract for error handling
    within the modular application.
    """

    def __init__(
        self,
        fallback_handler: Optional[FallbackHandler] = None,
        logging_level: int = logging.ERROR,
    ) -> None:
        """
        Initializes the ErrorHandler.

        :param fallback_handler: FallbackHandler instance to use for invoking fallback strategies.
                                 If ``None``, no fallback strategies will be invoked.
        :type fallback_handler: Optional[FallbackHandler], optional
        :param logging_level: Logging level for error handler operations and errors. Defaults to ``logging.ERROR``.
        :type logging_level: int
        """
        self.fallback_handler: Optional[FallbackHandler] = fallback_handler
        self.error_log: List[Dict[str, Any]] = (
            []
        )  # In-memory error log for tracking errors
        self.logging_level: int = logging_level
        self.logging = logging.getLogger(__name__)

    async def handle_error(
        self,
        task_id: str,  # Interface-required parameter: Unique ID of the task
        error: Exception,  # Interface-required parameter: The exception instance
        submodule_name: str,  # Interface-required parameter: Name of the submodule where the error occurred
        context: Dict[str, Any],  # Contextual information related to the error
        severity: str = "error",  # Interface-required parameter with default value
    ) -> Dict[str, Any]:
        """
        Interface-compliant error handling method.

        This method is the main entry point for handling errors within the application.
        It logs the error, attempts to execute a fallback strategy if configured,
        and returns a dictionary containing error information and potentially fallback results.

        :param task_id: The unique identifier of the task that encountered the error.
        :type task_id: str
        :param error: The exception object representing the error that occurred.
        :type error: Exception
        :param submodule_name: The name of the submodule where the error originated.
        :type submodule_name: str
        :param context: A dictionary containing contextual information about the error,
                        such as the operation being performed, relevant parameters, etc.
                        It can also include a 'fallback_strategy' key to suggest a fallback strategy name.
        :type context: Dict[str, Any]
        :param severity: Severity level of the error (e.g., 'error', 'critical', 'warning'). Defaults to 'error'.
        :type severity: str
        :return: A dictionary containing details about the error and any results from
                 fallback attempts. The dictionary structure may include keys like:
                 'task_id', 'error', 'submodule', 'severity', 'context',
                 and optionally 'fallback_result' or 'fallback_error'.
        :rtype: Dict[str, Any]
        :raises TypeError: if error is not an Exception or context is not a dictionary.
        """
        return await self._handle_error_impl(
            task_id, error, submodule_name, context, severity
        )

    async def _handle_error_impl(
        self,
        task_id: str,
        error: Exception,
        submodule_name: str,
        context: Dict[str, Any],
        severity: str,  # severity parameter added
    ) -> Dict[str, Any]:
        """
        Implementation of the error handling logic.

        This method encapsulates the core error handling steps: logging the error,
        attempting to execute a fallback strategy, and preparing the error information
        to be returned. It is called by the interface-compliant `handle_error` method.

        :param task_id: The unique identifier of the task that encountered the error.
        :type task_id: str
        :param error: The exception object representing the error.
        :type error: Exception
        :param submodule_name: The name of the submodule where the error originated.
        :type submodule_name: str
        :param context: A dictionary of contextual information about the error.
        :type context: Dict[str, Any]
        :param severity: Severity level of the error.
        :type severity: str
        :return: A dictionary containing error details and potential fallback results.
        :rtype: Dict[str, Any]
        """
        if not isinstance(error, Exception):
            raise TypeError(f"error must be an Exception, got {type(error).__name__}")
        if not isinstance(context, dict):
            raise TypeError(
                f"context must be a dictionary, got {type(context).__name__}"
            )

        # severity = context.get("severity", "error") # Severity now passed as parameter, no need to get from context with default

        # Log the error with context and severity
        self.log_error(error, context, severity, submodule_name, task_id)

        error_info = {
            "task_id": task_id,
            "error": str(error),
            "submodule": submodule_name,
            "severity": severity,
            "context": context,
        }

        if self.fallback_handler:
            fallback_strategy_name = context.get("fallback_strategy")
            try:
                fallback_result = await self.fallback_handler.execute_fallback(
                    task_id, strategy_name=fallback_strategy_name
                )
                if (
                    fallback_result
                ):  # Check if fallback returned a truthy value (not None, False, etc.)
                    self.logging.info(
                        f"Fallback strategy '{fallback_strategy_name}' executed successfully for task: {task_id} in submodule: {submodule_name}."
                    )
                    error_info["fallback_result"] = (
                        fallback_result  # Add fallback result to error info
                    )
                    return fallback_result  # Return fallback result if successful and meaningful
                else:
                    self.logging.warning(
                        f"Fallback strategy '{fallback_strategy_name}' for task: {task_id} in submodule: {submodule_name} did not return a result."
                    )
                    return error_info  # Return original error info if fallback didn't give a result
            except Exception as fallback_error:
                self.logging.error(
                    f"Fallback strategy '{fallback_strategy_name}' for task {task_id} in submodule: {submodule_name} failed with error: {fallback_error}",
                    exc_info=True,
                )
                error_info["fallback_error"] = str(
                    fallback_error
                )  # Record fallback error
                return error_info  # Return original error info even if fallback failed

        return error_info  # Return error info if no fallback handler or fallback not executed

    def log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: str,
        submodule_name: str,
        task_id: str,
    ) -> str:
        """
        Logs the error with detailed context, severity, submodule, and task ID.

        This method generates a detailed error message and logs it using the configured
        logging level. It includes contextual information, severity, submodule name,
        and task ID to provide comprehensive error tracking.

        :param error: The exception to log.
        :type error: Exception
        :param context: Contextual information related to the error.
        :type context: Dict[str, Any]
        :param severity: Severity level of the error.
        :type severity: str
        :param submodule_name: The name of the submodule where the error occurred.
        :type submodule_name: str
        :param task_id: The unique identifier of the task that encountered the error.
        :type task_id: str
        :return: A string message describing the logged error.
        :rtype: str
        """
        error_message = (
            f"Error in submodule: '{submodule_name}' for task: '{task_id}'. "
            f"Severity: '{severity}'. Context: {context}. Error: {error}"
        )
        log_level = getattr(
            logging, severity.upper(), logging.ERROR
        )  # Dynamically determine log level
        self.logging.log(log_level, error_message, exc_info=True)
        self.error_log.append(
            {
                "timestamp": datetime.now().isoformat(),
                "severity": severity,
                "submodule": submodule_name,
                "task_id": task_id,
                "error": str(error),
                "context": context,
            }
        )  # Append error to in-memory log
        return error_message

        # Removed redundant log_error method


#### Message Formatting and Handling Components

##### Message Formatters


class MessageFormatter(IMessageFormatter):  # Inherit from IMessageFormatter interface
    """
    Formats internal messages for consistent and stylised output.

    This class implements the :class:`IMessageFormatter` interface and provides
    configurable message formatting using Python format strings. It allows for
    standardizing the appearance of log messages and other internal communications.
    """

    def __init__(self, format_string: str = "Formatted Message: {message}") -> None:
        """
        Initializes the :class:`MessageFormatter`.

        :param format_string: The format string to use for formatting messages.
                              It should be a string that uses ``{message}`` as a placeholder
                              for the message to be formatted.  For example,
                              ``"Log: {message} - Timestamp: {timestamp}"``.
                              Defaults to "Formatted Message: {message}".
        :type format_string: str
        :raises TypeError: if ``format_string`` is not a string.
        """
        if not isinstance(format_string, str):
            raise TypeError(
                f"format_string must be a string, got {type(format_string)}"
            )
        self.format_string = format_string
        self.logging = logging.getLogger(__name__)

    def format_message(
        self, message: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Formats a message using the configured format string.

        This method attempts to format the given ``message`` using the format string
        provided during initialization. If formatting fails, typically due to
        a :exc:`KeyError` in the format string or other formatting errors,
        the original message is returned, and an error is logged.

        :param message: The message to format.
        :type message: str
        :param context: Optional context dictionary (not used in base formatter, but available for extension).
        :type context: Optional[Dict[str, Any]], optional
        :return: The formatted message. If formatting fails, returns the original message.
        :rtype: str
        """
        try:
            formatted_message = self.format_string.format(message=message)
            self.logging.debug(f"Formatted message: {message} -> {formatted_message}")
            return formatted_message
        except KeyError as e:
            self.logging.error(
                f"KeyError during message formatting due to invalid format string key: {e}. "
                "Falling back to raw message."
            )
            return message  # Fallback to raw message if formatting fails
        except Exception as e:
            self.logging.error(
                f"Unexpected error during message formatting: {e}. "
                "Falling back to raw message.",
                exc_info=True,
            )
            return message


##### Message Colorers


class MessageColorer(IMessageColorer):  # Inherit from IMessageColorer interface
    """
    Colours internal messages for consistent and stylised output using ANSI escape codes.

    This class implements the :class:`IMessageColorer` interface and provides
    configurable message coloring based on keywords and a color map. It enhances
    log readability by applying ANSI color codes to messages based on predefined
    keywords like 'error', 'warning', 'info', and 'debug'. Supports disabling color output
    for environments where ANSI codes are not supported or desired.
    """

    DEFAULT_COLOR_MAP: Dict[str, str] = {
        "error": "\033[91m",  # Red ANSI code
        "warning": "\033[93m",  # Yellow ANSI code
        "info": "\033[92m",  # Green ANSI code
        "debug": "\033[94m",  # Blue ANSI code
    }
    """Default color map for message coloring. Maps keywords to ANSI color codes."""

    RESET_CODE: str = "\033[0m"
    """ANSI reset code to reset text color to default."""

    def __init__(
        self,
        color_map: Optional[Dict[str, str]] = None,
        use_color: bool = True,
        enable_keyword_lookup: bool = True,
    ) -> None:
        """
        Initializes the :class:`MessageColorer`.

        :param color_map: Dictionary mapping message keywords to ANSI color codes.
                          If ``None``, defaults to :attr:`DEFAULT_COLOR_MAP`.
                          Keywords are case-insensitive when matching against messages.
                          Example: ``{"ALERT": "\\033[95m", "normal": "\\033[0m"}``.
        :type color_map: Optional[Dict[str, str]], optional
        :param use_color: Whether to enable color output. If ``False``, no coloring is applied,
                          and messages are returned as they are. Defaults to ``True``.
        :type use_color: bool, optional
        :param enable_keyword_lookup: Whether to enable keyword-based color lookup.
                                      If ``False``, no keyword lookup is performed, and messages are not colored
                                      based on keywords, unless colored explicitly by other means.
                                      Defaults to ``True``.
        :type enable_keyword_lookup: bool, optional
        """
        self.color_map = color_map if color_map is not None else self.DEFAULT_COLOR_MAP
        self.use_color = use_color
        self.enable_keyword_lookup = enable_keyword_lookup
        self.logging = logging.getLogger(__name__)

    def color_message(self, message: str, severity: str = "debug") -> str:
        """
        Colours a message based on keywords found in the message and the configured color map.

        This method iterates through the :attr:`color_map` keywords and checks if any keyword
        (case-insensitive) is present in the input ``message``. If a keyword is found,
        the corresponding ANSI color code is applied to the entire message, wrapping it
        with the color code and the :attr:`RESET_CODE`. If :attr:`use_color` is ``False``,
        or if :attr:`enable_keyword_lookup` is ``False``, or if no keyword is found in the message,
        the original message is returned without coloring. Only the first matching keyword's color is applied.

        :param message: The message to colour.
        :type message: str
        :param severity: Severity level of the message (defaults to 'debug', but not used in coloring logic).
        :type severity: str, optional
        :return: The coloured message. If no color is applied, returns the original message.
        :rtype: str
        """
        if not self.use_color:
            return message  # No coloring if use_color is False

        if not self.enable_keyword_lookup:
            return message  # No keyword lookup if disabled

        coloured_message = message
        for keyword, color_code in self.color_map.items():
            if keyword.lower() in message.lower():  # Case-insensitive keyword matching
                coloured_message = (
                    f"{color_code}{message}{self.RESET_CODE}"  # ANSI color code + reset
                )
                break  # Apply first matching color and stop looking
        self.logging.debug(f"Coloured message: {message} -> {coloured_message}")
        return coloured_message


#### Logging and Monitoring Core


class LoggingAndMonitoring:
    """
    Integrates and orchestrates various aspects of application monitoring.

    This class serves as a central hub for logging, resource monitoring, profiling,
    tracing, telemetry, and error handling. It promotes modularity by delegating
    specific tasks to dedicated components adhering to defined interfaces.

    **Components Integrated:**

    - :class:`IMessageFormatter`: Standardizes log message appearance.
    - :class:`IMessageColorer`: Enhances log readability with colors.
    - :class:`IResourceMonitor`: Tracks system resource usage.
    - :class:`IProfiler`: Analyzes task execution performance.
    - :class:`ITracingBackend`: Tracks execution flow across components.
    - :class:`ITelemetryCollector`: Gathers metrics for system behavior analysis.
    - :class:`IErrorHandler`: Manages and responds to application errors.
    """

    def __init__(
        self,
        message_formatter: Optional[IMessageFormatter] = None,
        message_colorer: Optional[IMessageColorer] = None,
        resource_monitor: Optional[IResourceMonitor] = None,
        profiler: Optional[IProfiler] = None,
        tracer: Optional[ITracingBackend] = None,
        telemetry_collector: Optional[ITelemetryCollector] = None,
        error_handler: Optional[IErrorHandler] = None,
    ) -> None:
        """
        Initializes the :class:`LoggingAndMonitoring` component with monitoring tools.

        If no component is provided for a specific aspect, a default implementation
        is instantiated to ensure basic functionality. This design allows for easy
        swapping of monitoring solutions by injecting custom implementations.

        :param message_formatter: Formatter for log messages. Defaults to :class:`MessageFormatter`.
        :type message_formatter: Optional[IMessageFormatter]
        :param message_colorer: Colorer for log messages. Defaults to :class:`MessageColorer`.
        :type message_colorer: Optional[IMessageColorer]
        :param resource_monitor: Monitor for system resources. Defaults to :class:`ResourceMonitor`.
        :type resource_monitor: Optional[IResourceMonitor]
        :param profiler: Profiler for task performance analysis. Defaults to :class:`Profiler`.
        :type profiler: Optional[IProfiler]
        :param tracer: Tracer for execution flow tracking. Defaults to :class:`Tracing` with :class:`InMemoryTracingBackend`.
        :type tracer: Optional[ITracingBackend]
        :param telemetry_collector: Collector for telemetry data. Defaults to :class:`TelemetryCollector`.
        :type telemetry_collector: Optional[ITelemetryCollector]
        :param error_handler: Handler for application errors. Defaults to :class:`ErrorHandler`.
        :type error_handler: Optional[IErrorHandler]

        :raises TypeError: if provided components do not adhere to their interfaces.
        """
        self.formatter: IMessageFormatter = message_formatter or MessageFormatter()
        self.colorer: IMessageColorer = message_colorer or MessageColorer()
        self.resource_monitor: IResourceMonitor = resource_monitor or ResourceMonitor()
        self.profiler: IProfiler = profiler or Profiler()
        self.tracer: ITracingBackend = tracer or Tracing()
        self.telemetry_collector: ITelemetryCollector = (
            telemetry_collector or TelemetryCollector()
        )
        self.error_handler: IErrorHandler = error_handler or ErrorHandler()

    ##### Logging Functionality

    def log_message(self, message: str, level: int = logging.INFO) -> None:
        """
        Logs a message, applying formatting and coloring if configured.

        The message is formatted and colored using the injected components before
        being logged through the standard Python logging library.

        :param message: The message string to be logged.
        :type message: str
        :param level: The logging level (e.g., logging.INFO, logging.DEBUG, logging.ERROR). Defaults to logging.INFO.
        :type level: int

        :raises TypeError: if message is not a string or level is not an integer.
        :raises ValueError: if level is not a valid logging level.
        """
        if not isinstance(message, str):
            raise TypeError(f"message must be a string, got {type(message).__name__}")
        if not isinstance(level, int):
            raise TypeError(f"level must be an integer, got {type(level).__name__}")
        if level not in [
            logging.CRITICAL,
            logging.ERROR,
            logging.WARNING,
            logging.INFO,
            logging.DEBUG,
            logging.NOTSET,
        ]:  # Basic validation of logging level
            raise ValueError(f"Invalid logging level: {level}")

        formatted_message = self.formatter.format_message(message)
        coloured_message = self.colorer.color_message(formatted_message)
        logging.log(level, coloured_message)

    ##### Resource Monitoring Functionality

    async def monitor_resources(self) -> Dict[str, float]:
        """
        Asynchronously monitors system resources.

        Delegates the task to the configured :class:`IResourceMonitor` and returns
        the collected resource metrics.

        :return: Dictionary of resource metrics (e.g., CPU, memory usage).
        :rtype: Dict[str, float]

        :raises ResourceMonitoringError: if resource monitoring fails.
        """
        return await self.resource_monitor.get_usage()

    ##### Profiling Functionality

    def profile_task(
        self,
        task: Callable[..., Any],
        context: Dict[str, Any],
        sort_by: str = "cumulative",
        task_id: Optional[str] = None,
        submodule_name: Optional[str] = None,
    ) -> Any:
        """
        Profiles the execution of a given task function.

        Utilizes the configured :class:`IProfiler` to profile the execution of the
        provided ``task`` function. Profiling results are typically saved to a report file.

        :param task: The function to be profiled.
        :type task: Callable[..., Any]
        :param context: Context information for the task.
        :type context: Dict[str, Any]
        :param sort_by: Sorting criterion for the profiling report (e.g., 'cumulative', 'tottime'). Defaults to 'cumulative'.
        :type sort_by: str, optional
        :param task_id: ID of the task being profiled. Optional.
        :type task_id: Optional[str]
        :param submodule_name: Name of the submodule where the task is profiled. Optional.
        :type submodule_name: Optional[str]
        :return: The result of the executed task function.
        :rtype: Any

        :raises ProfilingError: if profiling fails or ``sort_by`` is invalid.
        :raises TypeError: if task is not callable or context is not a dictionary.
        :raises ValueError: if sort_by is not a valid sorting option.
        """
        if not callable(task):
            raise TypeError(f"task must be callable, got {type(task).__name__}")
        if not isinstance(context, dict):
            raise TypeError(
                f"context must be a dictionary, got {type(context).__name__}"
            )

        updated_context = context.copy()
        updated_context.update({"task_id": task_id, "submodule_name": submodule_name})
        return self.profiler.start_profiling(context=updated_context)

    ##### Tracing Functionality

    def trace_event(
        self,
        task_id: str,
        message: str,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Traces a specific event with a message.

        Records a trace event using the injected :class:`ITracingBackend`. Useful for
        tracking execution flow and diagnosing issues.

        :param task_id: ID of the task or operation associated with the event.
        :type task_id: str
        :param message: Descriptive message for the trace event.
        :type message: str
        :param attributes: Additional attributes for the trace event. Optional.
        :type attributes: Optional[Dict[str, Any]], optional

        :raises TracingError: if tracing the event fails.
        :raises TypeError: if task_id or message are not strings.
        """
        if not isinstance(task_id, str):
            raise TypeError(f"task_id must be a string, got {type(task_id).__name__}")
        if not isinstance(message, str):
            raise TypeError(f"message must be a string, got {type(message).__name__}")
        self.tracer.trace(task_id, message, attributes=attributes)

    ##### Telemetry Functionality

    def start_telemetry_span(
        self,
        task_id: str,
        operation_name: str,
        context: Optional[Dict[str, Any]] = None,
        submodule_name: Optional[str] = None,
    ) -> "TelemetrySpan":
        """
        Starts a telemetry span.

        Begins a new telemetry span using the injected :class:`ITelemetryCollector`.
        Telemetry spans measure the duration and characteristics of operations.

        :param task_id: ID of the task or operation to start a span for.
        :type task_id: str
        :param operation_name: Name of the operation for the telemetry span.
        :type operation_name: str
        :param context: Context information for the telemetry span. Optional.
        :type context: Optional[Dict[str, Any]], optional
        :param submodule_name: Name of the submodule where the span is started. Optional.
        :type submodule_name: Optional[str]
        :return: A :class:`TelemetrySpan` object representing the started span.
        :rtype: TelemetrySpan

        :raises TelemetryError: if starting the telemetry span fails.
        :raises TypeError: if task_id is not a string.
        """
        if not isinstance(task_id, str):
            raise TypeError(f"task_id must be a string, got {type(task_id).__name__}")
        attributes: Dict[str, Any] = {}
        if context is not None:
            attributes.update(context)
        if submodule_name is not None:
            attributes["submodule_name"] = submodule_name
        return self.telemetry_collector.start_span(
            operation_name, task_id, attributes=attributes
        )

    def end_telemetry_span(
        self,
        task_id: str,
        context: Optional[Dict[str, Any]] = None,
        submodule_name: Optional[str] = None,
    ) -> None:
        """
        Ends a telemetry span.

        Terminates the telemetry span associated with ``task_id`` using :class:`ITelemetryCollector`.
        Ending a span records its duration and associated data.

        :param task_id: ID of the task or operation to end the telemetry span for.
        :type task_id: str
        :param context: Context information for the telemetry span. Optional.
        :type context: Optional[Dict[str, Any]], optional
        :param submodule_name: Name of the submodule where the span is ended. Optional.
        :type submodule_name: Optional[str]

        :raises TelemetryError: if ending the telemetry span fails.
        :raises TypeError: if task_id is not a string.
        """
        if not isinstance(task_id, str):
            raise TypeError(f"task_id must be a string, got {type(task_id).__name__}")
        attributes: Dict[str, Any] = {}
        if context is not None:
            attributes.update(context)
        if submodule_name is not None:
            attributes["submodule_name"] = submodule_name
        self.telemetry_collector.end_span(task_id, attributes=attributes)

    ##### Error Handling Functionality

    async def handle_error(
        self,
        task_id: str,
        submodule_name: str,
        error: Exception,
        context: Dict[str, Any],
        severity: str = "error",
    ) -> Dict[str, Any]:
        """
        Handles an error using the injected :class:`IErrorHandler`.

        Delegates error handling to the configured :class:`IErrorHandler`, providing
        contextual information about the error.

        :param task_id: ID of the task or operation where the error occurred.
        :type task_id: str
        :param submodule_name: Name of the submodule where the error occurred.
        :type submodule_name: str
        :param error: The exception object representing the error.
        :type error: Exception
        :param context: Context information about where the error occurred.
        :type context: Dict[str, Any]
        :param severity: Severity level of the error (e.g., 'error', 'critical', 'warning'). Defaults to 'error'.
        :type severity: str
        :return: Dictionary containing the result of error handling.
        :rtype: Dict[str, Any]

        :raises ErrorHandlingError: if the error handling process fails.
        :raises TypeError: if error is not an Exception.
        """
        if not isinstance(error, Exception):
            raise TypeError(f"error must be an Exception, got {type(error).__name__}")
        if not isinstance(context, dict):
            raise TypeError(
                f"context must be a dictionary, got {type(context).__name__}"
            )
        if not isinstance(severity, str):
            raise TypeError(f"severity must be a string, got {type(severity).__name__}")
        if not isinstance(task_id, str):
            raise TypeError(f"task_id must be a string, got {type(task_id).__name__}")
        if not isinstance(submodule_name, str):
            raise TypeError(
                f"submodule_name must be a string, got {type(submodule_name).__name__}"
            )

        return await self.error_handler.handle_error(
            task_id, error, submodule_name, context, severity
        )


#### Unified Logging and Monitoring Interface (Unilog)


class Unilog:
    """
    Unified logging and monitoring interface.

    Provides a single entry point for logging, resource monitoring, profiling, tracing,
    telemetry, metrics, and error handling. Simplifies integration of comprehensive
    monitoring by aggregating functionalities from :class:`LoggingAndMonitoring`
    and :class:`MetricsRegistry`.
    """

    def __init__(
        self,
        logging_monitoring: Optional[LoggingAndMonitoring] = None,
        metrics_registry: Optional["MetricsRegistry"] = None,
    ) -> None:
        """
        Initializes the Unilog wrapper.

        Creates default instances of :class:`LoggingAndMonitoring` and
        :class:`MetricsRegistry` if none are provided, allowing for immediate use
        with sensible defaults or custom component injection.

        :param logging_monitoring: Instance of :class:`LoggingAndMonitoring`. Defaults to a new instance if None.
        :type logging_monitoring: Optional[LoggingAndMonitoring]
        :param metrics_registry: Instance of :class:`MetricsRegistry`. Defaults to a new instance if None.
        :type metrics_registry: Optional[MetricsRegistry]
        """
        self.logging_monitoring: LoggingAndMonitoring = (
            logging_monitoring or LoggingAndMonitoring()
        )
        self.metrics_registry: "MetricsRegistry" = metrics_registry or MetricsRegistry()
        logging.info(
            "Unilog initialized with integrated logging and metrics components."
        )

    ##### LoggingAndMonitoring Methods

    def log_message(self, message: str, level: int = logging.INFO) -> None:
        """
        Logs a message using the integrated :class:`LoggingAndMonitoring` component.

        :inheritdoc: LoggingAndMonitoring.log_message
        """
        self.logging_monitoring.log_message(message, level)

    async def monitor_resources(self) -> Dict[str, float]:
        """
        Monitors system resources using the integrated :class:`LoggingAndMonitoring` component.

        :inheritdoc: LoggingAndMonitoring.monitor_resources
        """
        return await self.logging_monitoring.monitor_resources()

    def profile_task(
        self,
        task: Callable[..., Any],
        context: Dict[str, Any],
        sort_by: str = "cumulative",
    ) -> Any:
        """
        Profiles a task using the integrated :class:`LoggingAndMonitoring` component.

        :inheritdoc: LoggingAndMonitoring.profile_task
        """
        return self.logging_monitoring.profile_task(task, context, sort_by=sort_by)

    def trace_event(self, task_id: str, message: str) -> None:
        """
        Traces an event using the integrated :class:`LoggingAndMonitoring` component.

        :inheritdoc: LoggingAndMonitoring.trace_event
        """
        self.logging_monitoring.trace_event(task_id, message)

    def start_telemetry_span(
        self, task_id: str, operation_name: str
    ) -> "TelemetrySpan":
        """
        Starts a telemetry span using the integrated :class:`LoggingAndMonitoring` component.

        :inheritdoc: LoggingAndMonitoring.start_telemetry_span
        """
        return self.logging_monitoring.start_telemetry_span(operation_name, task_id)

    def end_telemetry_span(self, task_id: str) -> None:
        """
        Ends a telemetry span using the integrated :class:`LoggingAndMonitoring` component.

        :inheritdoc: LoggingAndMonitoring.end_telemetry_span
        """
        self.logging_monitoring.end_telemetry_span(task_id)

    async def handle_error(
        self,
        task_id: str,
        submodule_name: str,
        error: Exception,
        context: Dict[str, Any],
        severity: str = "error",
    ) -> Dict[str, Any]:
        """
        Handles an error using the integrated :class:`LoggingAndMonitoring` component.

        :inheritdoc: LoggingAndMonitoring.handle_error
        """
        return await self.logging_monitoring.handle_error(
            task_id, submodule_name, error, context, severity
        )

    ##### MetricsRegistry Methods

    def create_metric(
        self,
        name: str,
        metric_type: "MetricTypes",
        description: Optional[str] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> Any:
        """
        Creates a metric using the integrated :class:`MetricsRegistry` component.

        :inheritdoc: MetricsRegistry.create_metric
        """
        return self.metrics_registry.create_metric(
            name, metric_type, description=description, labels=labels
        )

    def increment_counter(
        self, name: str, value: int = 1, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Increments a counter metric using the integrated :class:`MetricsRegistry` component.

        :inheritdoc: MetricsRegistry.increment_counter
        """
        self.metrics_registry.increment_counter(name, value, labels=labels)

    def set_gauge(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Sets a gauge metric using the integrated :class:`MetricsRegistry` component.

        :inheritdoc: MetricsRegistry.set_gauge
        """
        self.metrics_registry.set_gauge(name, value, labels=labels)

    def observe_histogram(
        self, name: str, value: float, labels: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Observes a value in a histogram metric using the integrated :class:`MetricsRegistry` component.

        :inheritdoc: MetricsRegistry.observe_histogram
        """
        self.metrics_registry.observe_histogram(name, value, labels=labels)


# Model Pooling and Loading and Management


class ModelLoader:
    """
    Model Loader Component:
    Loads and manages different models based on model type and resource availability.

    This class is responsible for the actual loading of machine learning models.
    It decouples model loading logic from model selection and serving, providing
    a dedicated component for handling model loading operations.
    """

    def __init__(
        self,
        model_paths: Optional[Dict[str, str]] = None,
        default_model_type: str = "small",
    ):
        """
        Initializes ModelLoader with configuration parameters.

        :param model_paths: Dictionary mapping model types (e.g., "small", "medium") to their file paths.
                            If None, defaults to a predefined set of model paths.
        :type model_paths: Optional[Dict[str, str]]
        :param default_model_type: The model type to use as a fallback if a specific type is not found
                                  in `model_paths`. Defaults to "small".
        :type default_model_type: str
        """
        self.model_paths = (
            model_paths
            if model_paths
            else {
                "small": "/home/lloyd/Development/saved_models/Qwen/Qwen2.5-0.5B-Instruct",
                "medium": "/home/lloyd/Development/saved_models/Qwen/Qwen2.5-3.0B-Instruct",
                "large": "/home/lloyd/Development/saved_models/Qwen/Qwen2.5-7B-Instruct",
            }
        )
        self.default_model_type = default_model_type

    def load_model_for_type(self, model_type: str) -> Any:
        """
        Loads a model for a specific model type.

        This method attempts to load a model based on the provided model type.
        It first checks if a specific model path is configured for this type.
        If a path is found, it tries to load the model from that path.
        If no specific path is configured, it falls back to a default loading mechanism.

        :param model_type: The type of model to load (e.g., "small", "large", "specialized").
        :type model_type: str
        :raises ValueError: If the `model_type` is None or empty.
        :raises FileNotFoundError: If a model path is configured for the type but the file is not found.
        :raises Exception: For any other errors during model loading (implementation specific).
        :return: The loaded model instance. In this placeholder implementation, it returns a string
                 indicating the model type and path.
        :rtype: Any
        """
        if not model_type:
            raise ValueError("Model type cannot be None or empty.")

        model_path = self.model_paths.get(
            model_type, self.model_paths[self.default_model_type]
        )
        if model_path:
            try:
                # Placeholder for actual model loading from path
                print(f"Loading model from path: {model_path} for type: {model_type}")
                # In a real implementation, this would load the model from model_path
                # e.g., using torch.load(model_path) for PyTorch models
                return f"loaded-model-{model_type}-from-{model_path}"  # Placeholder return value
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Model file not found at path: {model_path} for type: {model_type}"
                )
            except Exception as e:
                raise Exception(
                    f"Error loading model of type {model_type} from {model_path}: {e}"
                )
        else:
            print(
                f"No specific path configured for model type: {model_type}. Using default loading."
            )
            # Placeholder for default model loading logic
            return f"default-model-{model_type}"  # Placeholder return value

    def load_optimal_model(self, current_mem: int, gpu_mem: int) -> Any:
        """
        Selects and loads the optimal model based on available system resources.

        This method determines the best model to load by considering the available CPU and GPU memory.
        It employs a heuristic to choose between different model types (e.g., "small", "medium", "large")
        based on memory thresholds. The specific logic for optimal model selection can be customized
        to fit different resource profiles and model requirements.

        :param current_mem: Available CPU memory in bytes.
        :type current_mem: int
        :param gpu_mem: Available GPU memory in bytes.
        :type gpu_mem: int
        :return: The loaded optimal model instance. In this placeholder, returns a string indicating
                 the selected model type.
        :rtype: Any
        """
        print(
            f"Loading optimal model based on memory: CPU {current_mem}, GPU {gpu_mem}"
        )

        # Heuristic for model selection based on memory availability
        if gpu_mem > 2 * (1024**3):  # Threshold for > 2GB GPU memory
            model_type = "large-model"
        elif current_mem > 4 * (1024**3):  # Threshold for > 4GB CPU memory
            model_type = "medium-model"
        else:
            model_type = "small-model"

        print(f"Selected model type: {model_type} based on resource availability.")
        return self.load_model_for_type(model_type)


class ModelFactory:
    """
    Model Factory Component:
    Factory for creating and managing ModelLoader instances.

    This class encapsulates the creation of ModelLoader instances, providing
    a centralized point for instantiating and configuring model loaders.
    It supports dependency injection and loose coupling by allowing different
    ModelLoader implementations to be easily managed and switched.
    """

    def __init__(
        self,
        model_loader_factory: Optional[Callable[[], ModelLoader]] = None,
        **loader_kwargs: Any,
    ):
        """
        Initializes ModelFactory with either a factory function or direct loader arguments.

        :param model_loader_factory: An optional callable (function or class) that, when called,
                                     returns a ModelLoader instance. If provided, this factory will be used
                                     to create the ModelLoader. Defaults to None.
        :type model_loader_factory: Optional[Callable[[], ModelLoader]]
        :param loader_kwargs: Keyword arguments to be passed to the ModelLoader constructor
                              if `model_loader_factory` is None. These arguments are used to instantiate
                              a default ModelLoader.
        :type loader_kwargs: Any
        """
        if model_loader_factory:
            self.model_loader = model_loader_factory()
        else:
            self.model_loader = ModelLoader(**loader_kwargs)

    def get_loader(self) -> ModelLoader:
        """
        Retrieves the ModelLoader instance managed by this factory.

        :return: The ModelLoader instance.
        :rtype: ModelLoader
        """
        return self.model_loader

    def load_model_for_type(self, model_type: str) -> Any:
        """
        Loads a model for a specific type using the associated ModelLoader.

        This method forwards the request to the underlying ModelLoader instance
        to load a model of the specified type.

        :param model_type: The type of model to load.
        :type model_type: str
        :return: The loaded model instance, as returned by the ModelLoader.
        :rtype: Any
        """
        return self.model_loader.load_model_for_type(model_type)

    def load_optimal_model(self, current_mem: int, gpu_mem: int) -> Any:
        """
        Loads the optimal model based on available resources using the associated ModelLoader.

        This method forwards the request to the underlying ModelLoader to select
        and load the optimal model based on the provided memory resources.

        :param current_mem: Available CPU memory in bytes.
        :type current_mem: int
        :param gpu_mem: Available GPU memory in bytes.
        :type gpu_mem: int
        :return: The loaded optimal model instance, as returned by the ModelLoader.
        :rtype: Any
        """
        return self.model_loader.load_optimal_model(current_mem, gpu_mem)

    def preload_models(self, model_types: List[str]) -> None:
        """
        Preloads models for specified module types using the associated ModelLoader.

        This method iterates through a list of model types and preloads each model
        using the underlying ModelLoader, which can improve initial response times.

        :param model_types: A list of model types to preload.
        :type model_types: List[str]
        :raises TypeError: if `model_types` is not a list.
        """
        if not isinstance(model_types, list):
            raise TypeError("model_types must be a list of strings.")
        for model_type in model_types:
            self.model_loader.load_model_for_type(model_type)


class ModelVersionManager:
    """
    Model Version Manager Component:
    Manages versions of models, allowing for registration and retrieval of specific versions.

    This class provides basic version control for machine learning models.
    It enables registering different versions of models and retrieving them by
    model type and version identifier. It supports simple version tracking and retrieval.
    Future enhancements could include rollback capabilities, A/B testing support,
    and more sophisticated versioning strategies.
    """

    def __init__(self) -> None:
        """
        Initializes ModelVersionManager.
        """
        self.model_versions: Dict[str, Dict[str, Any]] = {}
        print("ModelVersionManager initialized.")

    def register_version(self, model_type: str, version: str, model: Any) -> None:
        """
        Registers a specific version of a model for a given model type.

        If a version already exists for the model type, it will be overwritten
        with the new model instance.

        :param model_type: The type of the model (e.g., "chess-engine").
        :type model_type: str
        :param version: The version identifier (e.g., "v1.0", "revision-2").
        :type version: str
        :param model: The loaded model instance for this version.
        :type model: Any
        :raises ValueError: if `model_type` or `version` is None or empty.
        """
        if not model_type or not version:
            raise ValueError("Model type and version cannot be None or empty.")
        if model_type not in self.model_versions:
            self.model_versions[model_type] = {}
        self.model_versions[model_type][version] = model
        print(f"Registered version '{version}' for model type '{model_type}'.")

    def get_version(self, model_type: str, version: str) -> Optional[Any]:
        """
        Retrieves a specific version of a model for a given model type.

        :param model_type: The type of the model.
        :type model_type: str
        :param version: The version identifier.
        :type version: str
        :return: The loaded model instance for the specified version, or None if not found.
        :rtype: Optional[Any]
        :raises ValueError: if `model_type` or `version` is None or empty.
        """
        if not model_type or not version:
            raise ValueError("Model type and version cannot be None or empty.")
        if (
            model_type in self.model_versions
            and version in self.model_versions[model_type]
        ):
            return self.model_versions[model_type][version]
        else:
            print(f"Version '{version}' not found for model type '{model_type}'.")
            return None

    def get_latest_version(self, model_type: str) -> Optional[Any]:
        """
        Retrieves the latest registered version of a model for a given model type.

        In this simple implementation, 'latest' is determined by the most recently
        registered version (lexicographically last key). For more sophisticated
        versioning, consider using semantic versioning or timestamps to determine
        the actual latest version.

        :param model_type: The type of the model.
        :type model_type: str
        :return: The loaded model instance for the latest version, or None if no versions are registered.
        :rtype: Optional[Any]
        :raises ValueError: if `model_type` is None or empty.
        """
        if not model_type:
            raise ValueError("Model type cannot be None or empty.")
        if model_type in self.model_versions:
            versions = self.model_versions[model_type]
            if versions:
                # Simple heuristic: last registered version is considered latest
                latest_version = list(versions.keys())[
                    -1
                ]  # In a real scenario, determine 'latest' more intelligently
                return versions[latest_version]
        print(f"No versions registered for model type '{model_type}'.")
        return None


class ModelServingLayer:
    """
    Model Serving Layer Component:
    Optimized model serving layer with caching and adaptive model selection.

    This layer manages the serving of machine learning models, focusing on
    performance and resource efficiency. It includes caching of active models
    for quick access and adaptive model selection based on available system
    resources using a ModelFactory.
    """

    def __init__(self, model_factory: ModelFactory, max_cached_models: int = 5):
        """
        Initializes ModelServingLayer with a ModelFactory and cache configuration.

        :param model_factory: Factory to load models. This factory is used to obtain
                              ModelLoader instances and load models as needed.
        :type model_factory: ModelFactory
        :param max_cached_models: Maximum number of models to cache in the LRU cache.
                                  Defaults to 5. This parameter controls the memory usage
                                  of the serving layer by limiting the number of cached models.
        :type max_cached_models: int
        :raises TypeError: if `model_factory` is not an instance of ModelFactory.
        :raises ValueError: if `max_cached_models` is not a positive integer.
        """
        if not isinstance(model_factory, ModelFactory):
            raise TypeError("model_factory must be an instance of ModelFactory.")
        if not isinstance(max_cached_models, int) or max_cached_models <= 0:
            raise ValueError("max_cached_models must be a positive integer.")

        self.active_models = LRUCache(
            maxsize=max_cached_models
        )  # LRU cache for storing active models
        self.model_factory = model_factory
        print(
            f"ModelServingLayer initialized with ModelFactory, caching up to {max_cached_models} models."
        )

    def get_adaptive_model(self) -> Any:
        """
        Retrieves the optimal model based on current system resources, using caching.

        This method implements smart model selection by checking available system
        resources (CPU and GPU memory) and using the ModelFactory to load the
        most appropriate model. The loaded model is then cached using an LRU cache
        for subsequent requests, significantly improving response times for
        repeated model access under similar resource conditions.

        :return: The optimal loaded model instance, potentially retrieved from cache.
        :rtype: Any
        """
        import psutil  # Import here to avoid circular dependency if ModelServingLayer is imported elsewhere
        import torch  # Import here to avoid circular dependency if ModelServingLayer is imported elsewhere

        current_mem = psutil.virtual_memory().available
        gpu_mem = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0

        @self.active_models  # Apply LRU cache to the model retrieval function
        def _load_and_cache_model(
            current_mem_key, gpu_mem_key
        ):  # Keys for caching based on memory profile
            print(
                f"Cache miss for memory profile: CPU {current_mem_key}, GPU {gpu_mem_key}. Loading new model."
            )
            return self.model_factory.load_optimal_model(
                current_mem=current_mem_key, gpu_mem=gpu_mem_key
            )

        # Use memory values as keys for caching - a simplified approach for demonstration
        model = _load_and_cache_model(current_mem, gpu_mem)
        return model


# endregion Model Serving


# Tasks and Task Management


class Task:
    """
    Represents a unit of work to be processed within the system.

    A Task encapsulates all necessary information for a specific operation,
    including its unique identifier, payload, processing context, status, and results.
    It serves as the fundamental unit managed by the :class:`TaskManager`.

    :ivar task_id: Unique identifier for the task.
    :vartype task_id: str
    :ivar payload: Input data required to execute the task.
    :vartype payload: Dict[str, Any]
    :ivar task_type: Type of the task, derived from the payload. Defaults to None if not specified in payload.
    :vartype task_type: Optional[str]
    :ivar context: Contextual information for task processing, can be updated during task lifecycle.
    :vartype context: Dict[str, Any]
    :ivar status: Current status of the task, e.g., 'pending', 'running', 'completed', 'failed'. Defaults to 'pending'.
    :vartype status: str
    :ivar result: Result of the task execution, populated upon completion. Defaults to None.
    :vartype result: Optional[Any]
    :ivar creation_time: Timestamp of when the task was created.
    :vartype creation_time: float
    """

    def __init__(self, task_id: str, payload: Dict[str, Any]):
        """
        Initializes a new Task instance.

        :param task_id: Unique identifier for this task. Must be a string.
        :type task_id: str
        :param payload: Dictionary containing the task's input data.
                      It is expected to contain all necessary information for task execution.
        :type payload: Dict[str, Any]
        :raises TypeError: if ``task_id`` is not a string or ``payload`` is not a dictionary.
        """
        if not isinstance(task_id, str):
            raise TypeError(f"task_id must be a string, got {type(task_id)}")
        if not isinstance(payload, dict):
            raise TypeError(f"payload must be a dict, got {type(payload)}")

        self.task_id = task_id
        self.payload = payload
        self.task_type: Optional[str] = payload.get(
            "task_type"
        )  # Example: task type from payload
        self.context: Dict[str, Any] = {}  # Processing context
        self.status: str = "pending"
        self.result: Optional[Any] = None
        self.creation_time: float = time.time()


class TaskManager:
    """
    Manages the lifecycle of tasks, from creation and storage to retrieval, status updates, and completion.

    The TaskManager provides an interface for interacting with tasks, abstracting
    the underlying storage mechanism. It is designed to handle task creation, retrieval,
    status updates, and completion in a consistent and controlled manner.

    :ivar tasks: Dictionary to store tasks, with task IDs as keys and Task instances as values.
    :vartype tasks: Dict[str, Task]
    """

    def __init__(self) -> None:  # Add explicit return type annotation
        """
        Initializes a new TaskManager instance.

        Sets up the internal task storage. Currently, tasks are stored in memory
        using a dictionary.
        """
        self.tasks: Dict[str, Task] = {}

    def create_task(self, task_payload: Dict[str, Any]) -> Task:
        """
        Creates a new task and registers it with the TaskManager.

        Generates a unique task ID, instantiates a :class:`Task` object with the given payload,
        and stores it in the task registry.

        :param task_payload: Dictionary containing the payload for the task.
                             Must include all necessary information for task execution.
        :type task_payload: Dict[str, Any]
        :return: The newly created Task instance.
        :rtype: Task
        :raises TypeError: if ``task_payload`` is not a dictionary.
        """
        if not isinstance(task_payload, dict):
            raise TypeError(f"task_payload must be a dict, got {type(task_payload)}")
        task_id = str(uuid.uuid4())
        task = Task(task_id=task_id, payload=task_payload)
        self.tasks[task_id] = task
        return task

    def get_task(self, task_id: str) -> "Task":
        """
        Retrieves a task from the TaskManager by its unique ID.

        :param task_id: The unique identifier of the task to retrieve.
        :type task_id: str
        :return: The Task instance corresponding to the given task ID.
        :rtype: Task
        :raises TypeError: if ``task_id`` is not a string.
        :raises KeyError: if no task is found with the given ``task_id``.
        """
        if not isinstance(task_id, str):
            raise TypeError(f"task_id must be a string, got {type(task_id)}")
        task = self.tasks.get(task_id)
        if task is None:
            raise KeyError(f"Task with id '{task_id}' not found.")
        return task

    def update_task_status(self, task_id: str, status: str) -> None:
        """
        Updates the status of a task.

        :param task_id: The unique identifier of the task to update.
        :type task_id: str
        :param status: The new status to set for the task.
        :type status: str
        :raises TypeError: if ``task_id`` or ``status`` is not a string.
        :raises KeyError: if no task is found with the given ``task_id``.
        """
        if not isinstance(task_id, str):
            raise TypeError(f"task_id must be a string, got {type(task_id)}")
        if not isinstance(status, str):
            raise TypeError(f"status must be a string, got {type(status)}")

        task = self.get_task(task_id)  # Will raise KeyError if task not found
        task.status = status

    def complete_task(self, task_id: str, result: Any) -> None:
        """
        Marks a task as completed and stores its result.

        :param task_id: The unique identifier of the task to complete.
        :type task_id: str
        :param result: The result of the task execution. Can be of any type.
        :type result: Any
        :raises TypeError: if ``task_id`` is not a string.
        :raises KeyError: if no task is found with the given ``task_id``.
        """
        if not isinstance(task_id, str):
            raise TypeError(f"task_id must be a string, got {type(task_id)}")

        task = self.get_task(task_id)  # Will raise KeyError if task not found
        task.status = "completed"
        task.result = result


# Quality Gateways


class QualityControlGateway:
    """
    Validates incoming tasks based on predefined criteria.

    This gateway ensures that tasks meet the initial requirements before being processed
    further in the pipeline. It uses a list of validation functions to check various
    aspects of the task.

    :param validation_functions: A list of callable functions that take a :class:`Task` object as input and return a boolean.
                                 Each function represents a validation criterion. The task is considered valid only if all
                                 validation functions return ``True``. Defaults to a basic check for task type presence.
    :type validation_functions: Optional[List[Callable[['Task'], bool]]]
    """

    def __init__(
        self, validation_functions: Optional[List[Callable[["Task"], bool]]] = None
    ) -> None:
        self.validation_functions: List[Callable[["Task"], bool]] = (
            validation_functions
            if validation_functions is not None
            else [self._default_task_type_check]
        )

    def _default_task_type_check(self, task: "Task") -> bool:
        """
        Default validation function to check if the task has a task_type.

        :param task: The task to validate.
        :type task: Task
        :return: True if task_type is present, False otherwise.
        :rtype: bool
        """
        return bool(task.task_type)

    def validate_task(self, task: "Task") -> bool:
        """
        Validates the given task against all configured validation criteria.

        Iterates through the list of validation functions and executes each one on the task.
        The task is considered valid only if all validation functions return ``True``.

        :param task: The task object to validate.
        :type task: Task
        :return: ``True`` if the task is valid according to all criteria, ``False`` otherwise.
        :rtype: bool
        :raises TypeError: if ``task`` is not a :class:`Task` instance.
        """
        if not isinstance(task, Task):
            raise TypeError(f"Expected a Task object, but got {type(task)}")

        for validation_func in self.validation_functions:
            if not validation_func(task):
                return False  # Task fails validation if any function returns False
        return True  # Task is valid if all validation functions pass


class QualityAssuranceGateway:
    """
    Ensures tasks are processed to a high standard during the processing phase.

    This gateway focuses on maintaining quality throughout the task execution lifecycle.
    It can include checks and processes that are performed while the task is being processed,
    to proactively ensure quality.

    :param assurance_functions: A list of callable functions that take a :class:`Task` object as input and return a boolean.
                                Each function represents a quality assurance check or process.
                                The gateway logic will determine how these functions are used to ensure quality
                                (e.g., all must pass, any must pass, etc.). Currently, it's a placeholder.
    :type assurance_functions: Optional[List[Callable[['Task'], bool]]]
    """

    def __init__(
        self, assurance_functions: Optional[List[Callable[["Task"], bool]]] = None
    ) -> None:
        self.assurance_functions: List[Callable[["Task"], bool]] = (
            assurance_functions if assurance_functions is not None else []
        )
        if assurance_functions is None:
            print(
                "QualityAssuranceGateway initialized without specific assurance functions. "
                "Ensure to configure them for actual quality assurance."
            )

    def ensure_quality(self, task: "Task") -> bool:
        """
        Executes quality assurance processes for the given task.

        This method is intended to perform actions or checks that ensure the quality of task processing.
        The actual implementation and logic depend on the configured ``assurance_functions``.
        Currently, it serves as a placeholder and always returns ``True``.

        :param task: The task object to perform quality assurance on.
        :type task: Task
        :return: ``True`` if quality assurance is considered successful (placeholder), actual behavior depends on implementation.
        :rtype: bool
        :raises TypeError: if ``task`` is not a :class:`Task` instance.
        """
        if not isinstance(task, Task):
            raise TypeError(f"Expected a Task object, but got {type(task)}")

        if not self.assurance_functions:
            print(
                "No assurance functions configured for QualityAssuranceGateway. "
                "Returning True as default placeholder behavior."
            )
            return True  # No assurance functions, default to pass for now

        # Placeholder logic - in real implementation, execute assurance functions and determine result
        # Example: for assurance_func in self.assurance_functions: ...
        print(
            "QualityAssuranceGateway is executing placeholder quality assurance logic."
        )
        return True  # Placeholder - quality assurance logic


class QualityVerificationGateway:
    """
    Verifies the quality of processed tasks against defined standards.

    This gateway checks if the task processing output meets the required quality levels.
    It uses a list of verification functions to assess different aspects of the processed task.

    :param verification_functions: A list of callable functions that take a :class:`Task` object as input and return a boolean.
                                   Each function represents a verification criterion. The task is considered verified
                                   only if all verification functions return ``True``. Defaults to no verification, always passing.
    :type verification_functions: Optional[List[Callable[['Task'], bool]]]
    """

    def __init__(
        self, verification_functions: Optional[List[Callable[["Task"], bool]]] = None
    ) -> None:
        self.verification_functions: List[Callable[["Task"], bool]] = (
            verification_functions if verification_functions is not None else []
        )
        if verification_functions is None:
            print(
                "QualityVerificationGateway initialized without specific verification functions. "
                "Verification will always pass by default."
            )

    def verify_quality(self, task: "Task") -> bool:
        """
        Verifies the quality of the processed task using configured verification criteria.

        Iterates through the list of verification functions and executes each one on the task.
        The task is considered verified only if all verification functions return ``True``.
        If no verification functions are configured, it defaults to passing verification.

        :param task: The task object to verify the quality of.
        :type task: Task
        :return: ``True`` if the task passes all verification criteria, ``False`` otherwise.
        :rtype: bool
        :raises TypeError: if ``task`` is not a :class:`Task` instance.
        """
        if not isinstance(task, Task):
            raise TypeError(f"Expected a Task object, but got {type(task)}")

        if not self.verification_functions:
            print(
                "No verification functions configured for QualityVerificationGateway. "
                "Returning True as default pass-through behavior."
            )
            return True  # No verification functions, default to pass

        # Placeholder logic - in real implementation, execute verification functions and determine result
        # Example: for verification_func in self.verification_functions: ...
        print(
            "QualityVerificationGateway is executing placeholder quality verification logic."
        )
        return True  # Placeholder - quality verification logic


class QualityValidationGateway:
    """
    Validates the final output of task processing against high-level requirements or business rules.

    This gateway performs a final validation step to ensure that the processed task and its result
    meet the overall objectives and comply with business logic or high-level specifications.

    :param validation_functions: A list of callable functions that take a :class:`Task` object as input and return a boolean.
                                   Each function represents a high-level validation criterion. The task is considered
                                   finally validated only if all validation functions return ``True``. Defaults to no validation, always passing.
    :type validation_functions: Optional[List[Callable[['Task'], bool]]]
    """

    def __init__(
        self, validation_functions: Optional[List[Callable[["Task"], bool]]] = None
    ) -> None:
        self.validation_functions: List[Callable[["Task"], bool]] = (
            validation_functions if validation_functions is not None else []
        )
        if validation_functions is None:
            print(
                "QualityValidationGateway initialized without specific validation functions. "
                "Validation will always pass by default."
            )

    def validate_quality(self, task: "Task") -> bool:
        """
        Validates the quality of the processed task against high-level validation criteria.

        Iterates through the list of validation functions and executes each one on the task.
        The task is considered finally validated only if all validation functions return ``True``.
        If no validation functions are configured, it defaults to passing validation.

        :param task: The task object to validate the final quality of.
        :type task: Task
        :return: ``True`` if the task passes all final validation criteria, ``False`` otherwise.
        :rtype: bool
        :raises TypeError: if ``task`` is not a :class:`Task` instance.
        """
        if not isinstance(task, Task):
            raise TypeError(f"Expected a Task object, but got {type(task)}")

        if not self.validation_functions:
            print(
                "No validation functions configured for QualityValidationGateway. "
                "Returning True as default pass-through behavior."
            )
            return True  # No validation functions, default to pass

        # Placeholder logic - in real implementation, execute validation functions and determine result
        # Example: for validation_func in self.validation_functions: ...
        print(
            "QualityValidationGateway is executing placeholder quality validation logic."
        )
        return True  # Placeholder - quality validation logic


# Submodule Interfaces


class ContentExtractor(ISubModule):
    """
    Extracts content from various sources.
    Part of the TaskPipeline, processes the context to extract content.

    This module is responsible for identifying and extracting relevant content
    from the input context within a Task. The specific extraction method can be configured
    during initialization.

    :param extraction_method: The method to use for content extraction.
                               Defaults to 'default'. Can be customized to use different
                               extraction techniques (e.g., regex-based, NLP-based).
    :type extraction_method: str, optional
    """

    def __init__(self, extraction_method: str = "default") -> None:
        """
        Initializes the ContentExtractor with a specified extraction method.
        """
        self.extraction_method = extraction_method
        logging.info(
            f"ContentExtractor initialized with method: {self.extraction_method}"
        )
        self._stage = ContentExtractorStage()  # Instantiate the stage class

    async def process(self, task: Task) -> Task:
        """
        Processes the task to extract content from its context.

        This method performs content extraction using the imported ContentExtractorStage.

        :param task: The task object containing the context.
        :type task: Task
        :raises TypeError: if ``task`` is not a :class:`Task` instance.
        :return: The updated task with extracted content in its context.
        :rtype: Task
        """
        logging.info("Starting Content Extraction Stage")
        if not isinstance(task, Task):
            logging.error(f"Expected a Task object, got {type(task)}")
            raise TypeError(f"Expected a Task object, got {type(task)}")

        context = task.context
        if not isinstance(context, dict):
            logging.error(
                f"Expected context to be a dict within Task, got {type(context)}"
            )
            raise TypeError(
                f"Expected context to be a dict within Task, got {type(context)}"
            )

        # Use the imported ContentExtractorStage for processing
        task.context = await self._stage.process(context)
        logging.debug(f"Extracted content. Context updated by ContentExtractorStage.")
        return task


class CodeAnalyzer(ISubModule):
    """
    Analyzes code snippets.
    Part of the TaskPipeline, processes the context within a Task to analyze code.

    This module is designed to analyze code snippets found within the context of a Task.
    The type of analysis performed can be configured during initialization.

    :param analysis_type: The type of code analysis to perform.
                          Defaults to 'syntax_check'. Can be customized for different
                          analysis types (e.g., 'complexity', 'security').
    :type analysis_type: str, optional
    """

    def __init__(self, analysis_type: str = "syntax_check") -> None:
        """
        Initializes the CodeAnalyzer with a specified analysis type.
        """
        self.analysis_type = analysis_type
        logging.info(
            f"CodeAnalyzer initialized with analysis type: {self.analysis_type}"
        )
        self._stage = CodeAnalyzerStage()  # Instantiate the stage class

    async def process(self, task: Task) -> Task:
        """
        Processes the task to analyze code snippets within its context.

        This method performs code analysis using the imported CodeAnalyzerStage.

        :param task: The task object containing the context.
        :type task: Task
        :raises TypeError: if ``task`` is not a :class:`Task` instance.
        :return: The updated task with code analysis results in its context.
        :rtype: Task
        """
        logging.info("Starting Code Analysis Stage")
        if not isinstance(task, Task):
            logging.error(f"Expected a Task object, got {type(task)}")
            raise TypeError(f"Expected a Task object, got {type(task)}")

        context = task.context
        if not isinstance(context, dict):
            logging.error(
                f"Expected context to be a dict within Task, got {type(context)}"
            )
            raise TypeError(
                f"Expected context to be a dict within Task, got {type(context)}"
            )

        # Use the imported CodeAnalyzerStage for processing
        task.context = await self._stage.process(context)
        logging.debug(f"Code analysis complete. Context updated by CodeAnalyzerStage.")
        return task


class EmotionalAnalyzer(ISubModule):
    """
    Analyzes emotional tone in text.
    Part of the TaskPipeline, processes the context within a Task to analyze emotions.

    This module analyzes the emotional tone of text content within the context of a Task.
    The emotional analysis model can be configured during initialization.

    :param emotion_model: The model or method to use for emotional analysis.
                          Defaults to 'default_model'. Can be customized to use different
                          emotion detection models or lexicons.
    :type emotion_model: str, optional
    """

    def __init__(self, emotion_model: str = "default_model") -> None:
        """
        Initializes the EmotionalAnalyzer with a specified emotion model.
        """
        self.emotion_model = emotion_model
        logging.info(
            f"EmotionalAnalyzer initialized with emotion model: {self.emotion_model}"
        )
        self._stage = EmotionalAnalyzerStage()  # Instantiate the stage class

    async def process(self, task: Task) -> Task:
        """
        Processes the task to analyze emotional tone within its context.

        This method performs emotional analysis using the imported EmotionalAnalyzerStage.

        :param task: The task object containing the context.
        :type task: Task
        :raises TypeError: if ``task`` is not a :class:`Task` instance.
        :return: The updated task with emotional analysis results in its context.
        :rtype: Task
        """
        logging.info("Starting Emotional Analysis Stage")
        if not isinstance(task, Task):
            logging.error(f"Expected a Task object, got {type(task)}")
            raise TypeError(f"Expected a Task object, got {type(task)}")

        context = task.context
        if not isinstance(context, dict):
            logging.error(
                f"Expected context to be a dict within Task, got {type(context)}"
            )
            raise TypeError(
                f"Expected context to be a dict within Task, got {type(context)}"
            )

        # Use the imported EmotionalAnalyzerStage for processing
        task.context = await self._stage.process(context)
        logging.debug(
            f"Emotional analysis complete. Context updated by EmotionalAnalyzerStage."
        )
        return task


class GapAnalyzer(ISubModule):
    """
    Analyzes gaps or discrepancies.
    Part of the TaskPipeline, processes the context within a Task to analyze gaps.

    This module is responsible for identifying and analyzing gaps or discrepancies
    within the context of a Task. The gap detection method can be configured during initialization.

    :param gap_detection_method: The method to use for gap detection.
                                 Defaults to 'default_method'. Can be customized to use different
                                 gap analysis techniques (e.g., statistical, rule-based).
    :type gap_detection_method: str, optional
    """

    def __init__(self, gap_detection_method: str = "default_method") -> None:
        """
        Initializes the GapAnalyzer with a specified gap detection method.
        """
        self.gap_detection_method = gap_detection_method
        logging.info(
            f"GapAnalyzer initialized with detection method: {self.gap_detection_method}"
        )
        self._stage = GapAnalyzerStage()  # Instantiate the stage class

    async def process(self, task: Task) -> Task:
        """
        Processes the task to analyze gaps or discrepancies within its context.

        This method performs gap analysis using the imported GapAnalyzerStage.

        :param task: The task object containing the context.
        :type task: Task
        :raises TypeError: if ``task`` is not a :class:`Task` instance.
        :return: The updated task with gap analysis results in its context.
        :rtype: Task
        """
        logging.info("Starting Gap Analysis Stage")
        if not isinstance(task, Task):
            logging.error(f"Expected a Task object, got {type(task)}")
            raise TypeError(f"Expected a Task object, got {type(task)}")

        context = task.context
        if not isinstance(context, dict):
            logging.error(
                f"Expected context to be a dict within Task, got {type(context)}"
            )
            raise TypeError(
                f"Expected context to be a dict within Task, got {type(context)}"
            )

        # Use the imported GapAnalyzerStage for processing
        task.context = await self._stage.process(context)
        logging.debug(f"Gap analysis complete. Context updated by GapAnalyzerStage.")
        return task


class SummaryGenerator(ISubModule):
    """
    Generates summaries of text.
    Part of the TaskPipeline, processes the context within a Task to generate summaries.

    This module is responsible for generating summaries of text content within the context of a Task.
    The summary generation method can be configured during initialization.

    :param summary_method: The method to use for summary generation.
                           Defaults to 'default_method'. Can be customized to use different
                           summarization techniques (e.g., extractive, abstractive).
    :type summary_method: str, optional
    """

    def __init__(self, summary_method: str = "default_method") -> None:
        """
        Initializes the SummaryGenerator with a specified summary method.
        """
        self.summary_method = summary_method
        logging.info(
            f"SummaryGenerator initialized with summary method: {self.summary_method}"
        )
        self._stage = SummaryGeneratorStage()  # Instantiate the stage class

    async def process(self, task: Task) -> Task:
        """
        Processes the task to generate summaries from its context.

        This method performs summary generation using the imported SummaryGeneratorStage.

        :param task: The task object containing the context.
        :type task: Task
        :raises TypeError: if ``task`` is not a :class:`Task` instance.
        :return: The updated task with summary results in its context.
        :rtype: Task
        """
        logging.info("Starting Summary Generation Stage")
        if not isinstance(task, Task):
            logging.error(f"Expected a Task object, got {type(task)}")
            raise TypeError(f"Expected a Task object, got {type(task)}")

        context = task.context
        if not isinstance(context, dict):
            logging.error(
                f"Expected context to be a dict within Task, got {type(context)}"
            )
            raise TypeError(
                f"Expected context to be a dict within Task, got {type(context)}"
            )

        # Use the imported SummaryGeneratorStage for processing
        task.context = await self._stage.process(context)
        logging.debug(
            f"Summary generation complete. Context updated by SummaryGeneratorStage."
        )
        return task


# Submodule Management


class SubModuleAllocator:
    """
    Manages submodule lifecycle and resource allocation.

    Responsible for creating, managing, and providing access to submodules.
    Leverages resource managers like ModelPool and VectorCache for submodule dependencies.

    :ivar submodules: Weakly referenced dictionary of active submodules, keyed by their object ID.
    :vartype submodules: weakref.WeakValueDictionary
    :ivar model_pool: Manages available models.
    :vartype model_pool: ModelPoolInterface
    :ivar vector_cache: Manages vector cache for submodules.
    :vartype vector_cache: VectorCacheInterface
    :ivar metrics: Collects telemetry data across submodules.
    :vartype metrics: TelemetryCollectorInterface
    :ivar submodule_classes: Dictionary mapping submodule types to their implementation classes.
    :vartype submodule_classes: Dict[str, Type[BaseSubModule]]
    """

    def __init__(
        self,
        model_pool: ModelPoolInterface,
        vector_cache: VectorCacheInterface,
        telemetry_collector: TelemetryCollectorInterface,
        submodule_classes: Optional[Dict[str, Type["BaseSubModule"]]] = None,
    ) -> None:
        """
        Initializes SubModuleAllocator with resource managers and submodule class mappings.

        :param model_pool: The model pool to be used by submodules.
        :type model_pool: ModelPoolInterface
        :param vector_cache: The vector cache to be used by submodules.
        :type vector_cache: VectorCacheInterface
        :param telemetry_collector: The telemetry collector for submodules.
        :type telemetry_collector: TelemetryCollectorInterface
        :param submodule_classes: Optional dictionary mapping submodule types to classes.
                                  Defaults to a predefined set if not provided.
        :type submodule_classes: Optional[Dict[str, Type[BaseSubModule]]]
        """
        self.submodules: WeakValueDictionary = WeakValueDictionary()
        self.model_pool: ModelPoolInterface = model_pool
        self.vector_cache: VectorCacheInterface = vector_cache
        self.metrics: TelemetryCollectorInterface = telemetry_collector
        self.submodule_classes: Dict[str, Type["BaseSubModule"]] = (
            submodule_classes
            or {
                "research": ResearchSubModule,
                "codegen": CodegenSubModule,
                "analysis": AnalysisSubModule,
                "validation": ValidationSubModule,
                "optimization": OptimizationSubModule,
                "content_extraction": ContentExtractionSubModule,
                "code_analysis": CodeAnalysisSubModule,
                "emotional_analysis": EmotionalAnalysisSubModule,
                "gap_analysis": GapAnalysisSubModule,
                "summary_generation": SummaryGenerationSubModule,
            }
        )
        logging.info("SubModuleAllocator initialized.")

    def register_submodule_type(
        self, module_type: str, submodule_class: Type["BaseSubModule"]
    ) -> None:
        """
        Registers a new submodule type with the allocator.

        Allows for dynamic registration of submodule types, extending the system's capabilities.

        :param module_type: The string identifier for the submodule type (e.g., "custom_module").
        :type module_type: str
        :param submodule_class: The class implementing the submodule logic, inheriting from BaseSubModule.
        :type submodule_class: Type[BaseSubModule]
        :raises TypeError: If ``submodule_class`` is not a subclass of BaseSubModule.
        :raises ValueError: If ``module_type`` is already registered.
        """
        if not issubclass(submodule_class, BaseSubModule):
            raise TypeError(
                f"Submodule class must be a subclass of BaseSubModule, got {submodule_class}."
            )
        if module_type in self.submodule_classes:
            raise ValueError(f"Module type '{module_type}' is already registered.")
        self.submodule_classes[module_type] = submodule_class
        logging.info(f"Submodule type '{module_type}' registered.")

    def create_submodule(self, module_type: str) -> "BaseSubModule":
        """
        Creates and registers a submodule of the specified type.

        Instantiates a submodule based on its type and makes it available for use.

        :param module_type: The type of submodule to create (e.g., "research", "codegen").
        :type module_type: str
        :return: An instance of the created submodule.
        :rtype: BaseSubModule
        :raises ValueError: If the ``module_type`` is invalid or not registered.
        :raises Exception: If there is an error during submodule instantiation.
        """
        if not isinstance(module_type, str):
            module_type = str(module_type)

        if module_type not in self.submodule_classes:
            error_message = f"Invalid module type: {module_type}. Available types: {list(self.submodule_classes.keys())}"
            logging.error(error_message)
            raise ValueError(error_message)

        submodule_class = self.submodule_classes[module_type]
        try:
            submodule_instance = submodule_class(self)  # Pass allocator instance
            self.submodules[id(submodule_instance)] = submodule_instance  # Track weakly
            logging.info(f"Submodule of type '{module_type}' created and registered.")
            return submodule_instance
        except Exception as e:
            error_message = f"Error creating submodule of type '{module_type}': {e}"
            logging.exception(error_message)
            raise Exception(error_message) from e


class BaseSubModule(SubModuleInterface):
    """
    Abstract base class for all submodules, providing core functionalities.

    Subclasses inherit common resources and define specific processing pipelines.
    Ensures all submodules have access to essential components like model pool,
    vector cache, telemetry, task processing pipeline, and error handling.

    :ivar allocator: Reference to the SubModuleAllocator managing this submodule.
    :vartype allocator: SubModuleAllocator
    :ivar model: Adaptive model instance from the ModelPool.
    :vartype model: Any # Depends on ModelPoolInterface
    :ivar vector_db: Vector store instance from the VectorCache.
    :vartype vector_db: Any # Depends on VectorCacheInterface
    :ivar task_processor: Pipeline for processing tasks.
    :vartype task_processor: TaskPipelineInterface
    :ivar telemetry: Telemetry collector for monitoring submodule operations.
    :vartype telemetry: TelemetryCollectorInterface
    :ivar error_handler: Error handler for managing errors within the submodule.
    :vartype error_handler: ErrorHandlerInterface
    """

    def __init__(
        self,
        allocator: SubModuleAllocator,
        task_processor: Optional[TaskPipelineInterface] = None,
        error_handler: Optional[ErrorHandlerInterface] = None,
    ) -> None:
        """
        Initializes BaseSubModule with allocator and core resources.

        :param allocator: The allocator managing this submodule, providing access to resources.
        :type allocator: SubModuleAllocator
        :param task_processor: Optional task pipeline to use for processing. Defaults to a new TaskPipeline instance.
        :type task_processor: Optional[TaskPipelineInterface]
        :param error_handler: Optional error handler to use. Defaults to a new ErrorHandler instance.
        :type error_handler: Optional[ErrorHandlerInterface]
        """
        self.allocator: SubModuleAllocator = allocator
        self.model = self.allocator.model_pool.get_adaptive_model(model_type="qwen")
        self.vector_db = self.allocator.vector_cache.get_store()
        self.telemetry: TelemetryCollectorInterface = self.allocator.metrics
        self.task_processor: TaskPipelineInterface = task_processor or TaskPipeline()
        self.error_handler: ErrorHandlerInterface = error_handler or ErrorHandler()
        logging.debug(
            f"BaseSubModule initialized by allocator: {allocator.__class__.__name__}"
        )

    async def process(self, task: Task) -> Any:
        """
        Processes a task through the submodule's lifecycle.

        This method orchestrates the full task processing within the submodule,
        including telemetry tracking, pipeline execution, output validation,
        result packaging, and error handling.

        :param task: The task object to be processed.
        :type task: Task
        :return: The processed result or error information.
        :rtype: Any
        """
        with self.telemetry.start_span(
            task_id=task.task_id,  # Add missing argument
            operation_name=f"{self.__class__.__name__}.process_task_{task.task_id}",
        ):
            logging.info(
                f"Submodule '{self.__class__.__name__}' processing task: {task.task_id}"
            )
            try:
                processed_output = await self._execute_pipeline(task)
                self._validate_outputs(processed_output, task)
                result = self._package_results(processed_output, task)
                logging.info(
                    f"Task {task.task_id} processed successfully by '{self.__class__.__name__}'."
                )
                return result
            except Exception as e:
                logging.error(
                    f"Error processing task {task.task_id} in '{self.__class__.__name__}': {e}"
                )
                return await self.error_handler.handle_error(
                    task.task_id, e, self.__class__.__name__, task.context
                )

    async def _execute_pipeline(self, task: Task) -> Any:
        """
        Executes the submodule-specific processing pipeline.

        This method must be overridden by concrete subclasses to define the actual
        task processing logic. It represents the core functionality of each submodule.

        :param task: The task object to be processed.
        :type task: Task
        :return: The output from the processing pipeline.
        :rtype: Any
        :raises NotImplementedError: If the method is not overridden in a subclass.
        """
        raise NotImplementedError("_execute_pipeline must be implemented in subclass.")

    def _validate_outputs(self, processed_output: Any, task: Task) -> None:
        """
        Validates the output from the processing pipeline.

        This method should be overridden by subclasses to implement specific
        validation logic for the processed output. It ensures data integrity and quality.

        :param processed_output: The output from the _execute_pipeline method.
        :type processed_output: Any
        :param task: The original task object.
        :type task: Task
        """
        logging.debug(
            "Output validation step in BaseSubModule - No validation implemented."
        )
        pass  # Validation logic to be implemented in subclasses

    def _package_results(self, processed_output: Any, task: Task) -> Any:
        """
        Packages the results of task processing for returning.

        This method should be overridden by subclasses to format and package the
        processed output into a structured result, potentially including metadata.

        :param processed_output: The output from the _execute_pipeline method.
        :type processed_output: Any
        :param task: The original task object.
        :type task: Task
        :return: The packaged result.
        :rtype: Any
        """
        logging.debug(
            "Result packaging in BaseSubModule - Returning processed output directly."
        )
        return (
            processed_output  # Result packaging logic to be implemented in subclasses
        )


class ValidationSubModule(BaseSubModule, SubModuleInterface):
    """
    Submodule specialized for real-time output validation and quality control.

    Incorporates a ValidationEngine and FallbackHandler to ensure output quality
    and manage fallback strategies in case of validation failures.

    :ivar validator: Engine for validating outputs.
    :vartype validator: ValidationEngineInterface
    :ivar fallback_handler: Handler for implementing fallback strategies.
    :vartype fallback_handler: FallbackHandlerInterface
    """

    def __init__(
        self,
        allocator: SubModuleAllocator,
        validator: Optional[ValidationEngineInterface] = None,
        fallback_handler: Optional[FallbackHandlerInterface] = None,
        task_processor: Optional[TaskPipelineInterface] = None,
        error_handler: Optional[ErrorHandlerInterface] = None,
    ):
        super().__init__(allocator, task_processor, error_handler)
        self.validator: ValidationEngineInterface = validator or ValidationEngine()
        self.fallback_handler: FallbackHandlerInterface = (
            fallback_handler or FallbackHandler()
        )
        logging.info(
            f"ValidationSubModule initialized with validator: {self.validator.__class__.__name__} and fallback handler: {self.fallback_handler.__class__.__name__}"
        )
        self._stage = ValidationStage()
        """Base class-compliant validation signature"""
        logging.debug("Executing validation checks in ValidationSubModule.")
        if not self._is_valid(processed_output, task):
            task_id = task.task_id
            logging.warning(
                f"Validation failed for task {task_id}. Raising ValidationError."
            )
            raise ValueError(f"Output validation failed for task {task_id}")
        logging.debug("Output validation passed in ValidationSubModule.")

    async def _execute_pipeline(self, task: Task) -> Any:
        """Validation pipeline with task context"""
        logging.info(f"Validating task {task.task_id} outputs")
        # primary_output = await super()._execute_pipeline(task) # No need to call super in this case as base is abstract
        task.context = await self._stage.process(
            task.context
        )  # Use ValidationStagePlaceholder
        self._validate_outputs(
            task.context, task
        )  # Pass task to validation, now validating context
        return await self._generate_validation_report(task)  # New async method

    async def _generate_validation_report(self, task: Task) -> Dict[str, Any]:
        """Generate detailed validation report"""
        return {
            "validation_result": "pass",  # This will be overwritten by ValidationStagePlaceholder if it modifies context
            "task_id": task.task_id,
            "checked_components": list(task.context.keys()),
            **task.context.get(
                "validation_result", {}
            ),  # Merge validation result from stage if available
        }

    def _is_valid(self, processed_output: Any, task: Task) -> bool:
        """Internal method to encapsulate validation logic."""
        if task is None:
            logging.error("Task context is not available for validation.")
            return False  # or raise an exception if task context is essential

        return self.validator.validate(processed_output, task.task_id, task.context)


class OptimizationSubModule(BaseSubModule, SubModuleInterface):
    """
    Submodule specialized for performance optimization and resource tuning.

    Incorporates ModelOptimizer and BenchmarkSuite to optimize model parameters
    and configurations for enhanced performance and resource utilization.

    :ivar optimizer: Optimizer for model parameters.
    :vartype optimizer: ModelOptimizerInterface
    :ivar benchmark_suite: Suite for benchmarking and finding optimal configurations.
    :vartype benchmark_suite: BenchmarkSuiteInterface
    """

    def __init__(
        self,
        allocator: SubModuleAllocator,
        optimizer: Optional[ModelOptimizerInterface] = None,
        benchmark_suite: Optional[BenchmarkSuiteInterface] = None,
        task_processor: Optional[TaskPipelineInterface] = None,
        error_handler: Optional[ErrorHandlerInterface] = None,
    ) -> None:
        """
        Initializes OptimizationSubModule with optimizer and benchmark suite.

        :param allocator: The allocator managing this submodule.
        :type allocator: SubModuleAllocator
        :param optimizer: Optional model optimizer to use. Defaults to a new ModelOptimizer instance.
        :type optimizer: Optional[ModelOptimizerInterface]
        :param benchmark_suite: Optional benchmark suite to use. Defaults to a new BenchmarkSuite instance.
        :type benchmark_suite: Optional[BenchmarkSuiteInterface]
        :param task_processor: Optional task pipeline to use for processing. Inherited from BaseSubModule.
        :type task_processor: Optional[TaskPipelineInterface]
        :param error_handler: Optional error handler to use. Inherited from BaseSubModule.
        :type error_handler: Optional[ErrorHandlerInterface]
        """
        super().__init__(allocator, task_processor, error_handler)
        self.optimizer: ModelOptimizerInterface = (
            optimizer or self.allocator.model_pool.get_optimizer()
        )  # Use pool reference
        self.benchmark_suite: BenchmarkSuiteInterface = benchmark_suite or BenchmarkSuite()  # type: ignore[name-defined]
        logging.info(
            f"OptimizationSubModule initialized with optimizer: {self.optimizer.__class__.__name__} and benchmark suite: {self.benchmark_suite.__class__.__name__}"
        )
        self._stage = OptimizationStage()  # Instantiate the stage class

    async def tune_parameters(self, task: Task) -> Any:
        """
        Tunes model parameters for optimal performance using benchmarking.

        Uses BenchmarkSuite to find the optimal configuration for the given task
        and applies it using ModelOptimizer.

        :param task: The task object for which parameters are to be tuned.
        :type task: Task
        :return: The result of applying optimized parameters.
        :rtype: Any
        """
        logging.info(
            f"OptimizationSubModule tuning parameters for task: {task.task_id}"
        )
        optimized_config = await self.benchmark_suite.find_optimal_config(
            task, task.context
        )
        if optimized_config:
            logging.info(
                f"Optimal configuration found, applying optimizer for task: {task.task_id}"
            )
            return await self.optimizer.apply(optimized_config, task.context)
        else:
            logging.warning(f"No optimal configuration found for task {task.task_id}.")
            return None  # Or raise an exception, depending on desired behavior

    async def _execute_pipeline(self, task: Task) -> Any:
        """
        Executes the optimization pipeline.

        Currently, this submodule focuses on parameter tuning and includes OptimizationStagePlaceholder.

        :param task: The task object.
        :type task: Task
        :return: The result of the optimization process.
        :rtype: Any
        """
        logging.debug(
            f"OptimizationSubModule executing pipeline (tuning parameters and stage placeholder) for task: {task.task_id}"
        )
        # First tune parameters
        tuning_result = await self.tune_parameters(task)
        # Then execute the placeholder stage
        task.context = await self._stage.process(
            task.context
        )  # Use OptimizationStagePlaceholder
        return {
            "tuning_result": tuning_result,
            "stage_result": task.context.get(
                "optimization_result", "Optimization stage executed"
            ),  # Capture stage result
        }


class ResearchSubModule(BaseSubModule):
    """Submodule for research-oriented tasks"""

    def __init__(
        self,
        allocator: SubModuleAllocator,
        task_processor: Optional[TaskPipelineInterface] = None,
        error_handler: Optional[ErrorHandlerInterface] = None,
    ):
        super().__init__(allocator, task_processor, error_handler)
        self._content_extractor_stage = ContentExtractorStage()
        self._summary_generator_stage = SummaryGeneratorStage()

    async def _execute_pipeline(self, task: Task) -> Any:
        context = task.context
        context = await self._content_extractor_stage.process(context)
        context = await self._summary_generator_stage.process(context)
        task.context = context
        return task.context


class CodegenSubModule(BaseSubModule):
    """Submodule for code generation tasks"""

    def __init__(
        self,
        allocator: SubModuleAllocator,
        task_processor: Optional[TaskPipelineInterface] = None,
        error_handler: Optional[ErrorHandlerInterface] = None,
    ):
        super().__init__(allocator, task_processor, error_handler)
        self._content_extractor_stage = ContentExtractorStage()
        self._code_analyzer_stage = CodeAnalyzerStage()

    async def _execute_pipeline(self, task: Task) -> Any:
        context = task.context
        context = await self._content_extractor_stage.process(context)
        context = await self._code_analyzer_stage.process(context)
        task.context = context
        return task.context


class AnalysisSubModule(BaseSubModule):
    """Submodule for data analysis tasks"""

    def __init__(
        self,
        allocator: SubModuleAllocator,
        task_processor: Optional[TaskPipelineInterface] = None,
        error_handler: Optional[ErrorHandlerInterface] = None,
    ):
        super().__init__(allocator, task_processor, error_handler)
        self._content_extractor_stage = ContentExtractorStage()
        self._emotional_analyzer_stage = EmotionalAnalyzerStage()
        self._gap_analyzer_stage = GapAnalyzerStage()

    async def _execute_pipeline(self, task: Task) -> Any:
        context = task.context
        context = await self._content_extractor_stage.process(context)
        context = await self._emotional_analyzer_stage.process(context)
        context = await self._gap_analyzer_stage.process(context)
        task.context = context
        return task.context


# Task Routing and Orchestration
class RoutingRuleBook:
    """Rule book for routing tasks to modules."""

    def select_module(self, task: "Task") -> str:
        """Selects a module type based on the task type."""
        task_type = task.task_type
        if task_type == "research":
            return "research"
        elif task_type == "codegen":
            return "codegen"
        elif task_type == "analysis":
            return "analysis"
        elif task_type == "validation":
            return "validation"
        elif task_type == "optimization":
            return "optimization"
        else:
            return "default"  # Default module type


class WorkflowCoordinator:
    """Pure workflow logic"""

    def __init__(self):
        """Initializes WorkflowCoordinator with workflow engine, quality gate, and task manager."""
        self.workflow_engine = WorkflowEngine()  # Manages task workflows and routing
        self.quality_gate = (
            QualityGateway()
        )  # Gate for quality checks on incoming tasks (defined elsewhere)
        self.task_manager = TaskManager()  # Manages tasks lifecycle (defined elsewhere)

    def distribute_task(
        self, task_payload: dict, load_balancer: "LoadBalancer"
    ):  # Accepts task payload as dict and LoadBalancer instance
        """Intelligent task routing with quality checks and delegates submodule creation to LoadBalancer.
        Creates a Task object from payload, validates it, routes it to a submodule via LoadBalancer,
        and processes the task.
        Args:
            task_payload (dict): The payload containing task information.
            load_balancer (LoadBalancer): The LoadBalancer instance to allocate submodules.
        Returns:
            Any: The result of task processing from the submodule.
        Raises:
            InvalidTaskError: If the task fails quality validation.
        """
        task = self.task_manager.create_task(
            task_payload
        )  # Create Task object from payload using TaskManager
        if self.quality_gate.validate_task(
            task
        ):  # Validate the task using QualityGateway
            # Get module type string instead of pipeline
            module_type = self.workflow_engine.route_task(task)
            submodule = load_balancer.create_submodule_for_task(module_type)
            return submodule.process(task)
        raise InvalidTaskError(f"Rejected task: {task.task_id}")


class LoadBalancer:
    """Resource-aware task distribution"""

    def __init__(self):
        """Initializes LoadBalancer with resource allocators and managers."""
        # Initialize required dependencies
        self.model_pool = ModelPool()
        self.vector_cache = VectorCache()
        self.telemetry_collector = TelemetryCollector()

        # Create allocator with required dependencies
        self.allocator = SubModuleAllocator(
            model_pool=self.model_pool,
            vector_cache=self.vector_cache,
            telemetry_collector=self.telemetry_collector,
        )
        self.resource_monitor = ResourceMonitor()

    async def initialize(self):
        """Cold start initialization with resource pre-warming.
        Establishes resource baselines and preloads models into the ModelPool.
        """
        await self.resource_monitor.establish_baselines()  # Establish initial resource baselines
        self.allocator.model_pool.preload_models(
            ["research", "codegen", "analysis"]
        )  # Preload models for common module types

    def create_submodule_for_task(self, target_module: str):
        """Creates submodule instance using SubModuleAllocator based on resource availability.
        Args:
            target_module (str): The module type to create.
        Returns:
            ISubModule: An instance of the requested submodule.
        """
        submodule = self.allocator.create_submodule(
            target_module
        )  # Create submodule instance using SubModuleAllocator
        return submodule


class ClusterStateManager:
    """Distributed state management"""

    def __init__(self):
        """Initializes ClusterStateManager for distributed state management."""
        pass

    # Add methods for cluster state management as needed


class PipelineTemplates:
    """Stores predefined pipeline templates for different task types.
    Manages a collection of pipeline templates, loaded or defined at runtime.
    """

    def __init__(self):
        """Initializes PipelineTemplates with an empty collection of templates."""
        self.templates: Dict[str, Dict[str, list]] = (
            {}
        )  # Dictionary to store pipeline templates, e.g., by task type

    def register_template(self, template_id: str, template_config: Dict[str, list]):
        """Registers a new pipeline template.
        Args:
            template_id (str): Unique identifier for the template.
            template_config (Dict[str, list]): Configuration of the pipeline template,
                                              e.g., stages and their order.
        """
        self.templates[template_id] = template_config

    def get_template(self, template_id: str) -> Dict[str, list]:
        """Retrieves a pipeline template by its ID.
        Args:
            template_id (str): The ID of the template to retrieve.
        Returns:
            Dict[str, list]: The pipeline template configuration, or None if not found.
        """
        return self.templates.get(template_id)

    def has_template(self, template_id: str) -> bool:
        """Checks if a template with the given ID exists.
        Args:
            template_id (str): The ID of the template to check.
        Returns:
            bool: True if the template exists, False otherwise.
        """
        return template_id in self.templates


class PipelineBuilder:
    """Declarative pipeline construction.
    Builds task pipelines dynamically based on templates and configurations.
    """

    def __init__(self, templates: PipelineTemplates):
        """Initializes PipelineBuilder with access to pipeline templates.
        Args:
            templates (PipelineTemplates): An instance of PipelineTemplates to retrieve templates from.
        """
        self.templates = templates
        self.stage_classes = (
            {  # Define stage_classes here within PipelineBuilder for clarity and access
                "ContentExtractor": ContentExtractor,
                "CodeAnalyzer": CodeAnalyzer,
                "EmotionalAnalyzer": EmotionalAnalyzer,
                "GapAnalyzer": GapAnalyzer,
                "SummaryGenerator": SummaryGenerator,
                "ValidationStagePlaceholder": ValidationStagePlaceholder,
                "OptimizationStagePlaceholder": OptimizationStagePlaceholder,
            }
        )

    def build_pipeline(self, template_id: str) -> List[Any]:
        """Constructs a pipeline (list of stages) from a template.
        Args:
            template_id (str): ID of the pipeline template to build.
        Returns:
            list: A list of instantiated pipeline stages based on the template, or None if template not found.
        """
        template_config = self.templates.get_template(template_id)
        if not template_config:
            return None  # Template not found

        pipeline_stages = []

        for stage_name in template_config.get(
            "stages", []
        ):  # Assuming template has 'stages' list
            stage_class = self.stage_classes.get(stage_name)
            if stage_class:
                pipeline_stages.append(
                    stage_class()
                )  # Instantiate stage and add to pipeline
            else:
                print(
                    f"Warning: Stage class '{stage_name}' not found."
                )  # Handle missing stage class

        return pipeline_stages


class PipelineOptimizer:
    """Runtime pipeline optimization.
    Optimizes a given pipeline based on runtime conditions and task characteristics.
    """

    def optimize_pipeline(self, pipeline: List[Any], task: "Task") -> List[Any]:
        """Optimizes the pipeline based on task and system conditions.
        Args:
            pipeline (list): The pipeline (list of stages) to optimize.
            task (Task): The task object for which the pipeline is being optimized.
        Returns:
            list: The optimized pipeline (potentially reordered or modified stages).
        """
        print("Applying runtime pipeline optimization logic.")
        # Example optimization: Reorder pipeline based on task type priority
        stage_priority = {
            "ContentExtractor": 2,
            "CodeAnalyzer": 3,
            "EmotionalAnalyzer": 1,
            "GapAnalyzer": 2,
            "SummaryGenerator": 4,
            "ValidationStagePlaceholder": 5,
            "OptimizationStagePlaceholder": 1,
        }  # Higher number means higher priority (executed later)

        def get_stage_order(stage_instance):
            stage_name = stage_instance.__class__.__name__
            return stage_priority.get(stage_name, 0)  # Default priority 0 if not found

        optimized_pipeline = sorted(
            pipeline, key=get_stage_order
        )  # Sort stages by priority

        print("Pipeline stages reordered based on priority.")
        return optimized_pipeline  # Return the optimized pipeline


class WorkflowEngine:
    """Intelligent task routing and pipeline management.
    Manages routing rules, pipeline templates, and dynamic pipeline creation and optimization.
    """

    def __init__(self):
        """Initializes WorkflowEngine with routing rules, pipeline templates, builder, and optimizer."""
        self.routing_rules = RoutingRuleBook()  # Rule book for routing tasks to modules
        self.pipeline_templates = PipelineTemplates()  # Manages pipeline templates
        self.pipeline_builder = PipelineBuilder(
            self.pipeline_templates
        )  # Builds pipelines from templates
        self.pipeline_optimizer = PipelineOptimizer()  # Optimizes pipelines at runtime

        # Add valid submodule classes reference
        self.submodule_classes = [
            "research",
            "codegen",
            "analysis",
            "validation",
            "optimization",
        ]

        # Existing template registration
        self.pipeline_templates.register_template(
            "default_pipeline",
            {"stages": ["ContentExtractor", "CodeAnalyzer", "SummaryGenerator"]},
        )
        self.pipeline_templates.register_template(
            "research_pipeline",
            {"stages": ["ContentExtractor", "GapAnalyzer", "SummaryGenerator"]},
        )
        self.pipeline_templates.register_template(
            "codegen_pipeline",
            {"stages": ["CodeAnalyzer", "SummaryGenerator"]},
        )
        self.pipeline_templates.register_template(
            "analysis_pipeline",
            {"stages": ["EmotionalAnalyzer", "SummaryGenerator"]},
        )
        self.pipeline_templates.register_template(
            "validation_pipeline",
            {"stages": ["ValidationStagePlaceholder"]},
        )
        self.pipeline_templates.register_template(
            "optimization_pipeline",
            {"stages": ["OptimizationStagePlaceholder"]},
        )

    def route_task(self, task: "Task") -> List[Any]:
        """Routes the task to a pipeline based on task type and routing rules.
        Returns:
            list: A list of pipeline stages (instances of processing classes).
        """
        module_type = self.routing_rules.select_module(task)
        if module_type not in self.submodule_classes:
            raise ValueError(f"Invalid routed module type: {module_type}")

        # Map module_type to pipeline template ID
        template_id = f"{module_type}_pipeline"
        if not self.pipeline_templates.has_template(template_id):
            template_id = (
                "default_pipeline"  # Fallback to default if no specific template
            )

        pipeline = self.pipeline_builder.build_pipeline(template_id)
        if pipeline is None:
            raise ValueError(f"No pipeline found for template ID: {template_id}")

        optimized_pipeline = self.pipeline_optimizer.optimize_pipeline(
            pipeline, task
        )  # Apply runtime optimization

        return optimized_pipeline

    def get_system_load(self):
        """Simulated system load (replace with actual metrics).
        Placeholder for a method to retrieve or calculate the current system load.
        Returns:
            float: A simulated system load value (0.0 to 1.0).
        """
        return 0.5  # Placeholder for actual load calculation, should be replaced with real metrics


class TaskPipeline(TaskPipelineInterface):
    """Executes a dynamically built pipeline for task processing."""

    async def execute(self, pipeline: List[Any], task: "Task") -> Dict[str, Any]:
        """Executes a given pipeline (list of stages) sequentially on the task context.
        Args:
            pipeline (list): List of pipeline stages (instances of processing classes).
            task (Task): The task object whose context will be processed.
        Returns:
            Dict[str, Any]: The final context after processing through all stages.
        """
        context = task.context.copy()
        for stage in pipeline:
            context = await stage.process(context)
        return context

    async def process(self, task: "Task") -> "Task":
        """Processes a task using a dynamically determined and executed pipeline.
        Args:
            task (Task): The task object to be processed.
        Returns:
            Task: The task object with updated context after pipeline execution.
        """
        workflow_engine = (
            WorkflowEngine()
        )  # Consider dependency injection for WorkflowEngine
        pipeline = workflow_engine.route_task(
            task
        )  # Route task to get the pipeline from WorkflowEngine
        context = await self.execute(pipeline, task)  # Now pipeline is a list of stages
        task.context = context
        return task


# Knowledge Management


class VectorDBManager:
    """Enhanced with original storage logic.
    Manages vector database operations, including storing and retrieving vector embeddings.
    """

    def store(self, documents):
        """Incorporates FAISS logic from original _store_documents.
        Placeholder for storing documents and their vector embeddings in the vector database.
        Args:
            documents: The documents to be stored (format depends on vector DB implementation).
        """
        if not hasattr(self, "vector_db"):  # Lazy initialization of vector_db
            self._initialize_vector_store()  # Initialize vector store if not already initialized

        # Original FAISS handling code - Placeholder for actual vector DB interaction
        self.vector_db.add_documents(documents)  # Add documents to vector DB
        self._atomic_save()  # Atomically save the vector DB state


class VectorCache:
    """Manages a cache for vector embeddings to improve performance.
    Provides access to a VectorDBManager instance, potentially using caching mechanisms.
    """

    def get_store(self):
        """Gets a VectorDBManager instance, potentially from a cache.
        For now, it returns a new instance each time. Caching logic can be added here.
        Returns:
            VectorDBManager: An instance of VectorDBManager.
        """
        print(
            "Getting vector store from cache"
        )  # Placeholder for cache retrieval logic
        return (
            VectorDBManager()
        )  # Returning a new instance for simplicity, caching can be added here


class GraphManager:
    """Knowledge graph operations"""

    def create_entity(self):
        pass


class ExplainabilityEngine:
    """Model decision explanations"""

    def generate_shap_values(self):
        pass


# Web Search Management


class WebSearchHandler:
    """
    Handles web search operations using SearxNG, content extraction with Tika,
    and basic web crawling to manage and process web-based knowledge.
    """

    def __init__(self):
        """
        Initializes the WebSearchHandler, loads processed URLs to avoid duplicates,
        and ensures the document store directory exists.
        """
        self.processed_urls = self._load_processed_urls()
        os.makedirs(DOCUMENT_STORE, exist_ok=True)

    def _load_processed_urls(self):
        """
        Loads the set of processed URLs from the JSON log file.
        """
        if os.path.exists(PROCESSED_LOG):
            with open(PROCESSED_LOG, "r") as f:
                return set(json.load(f))
        return set()

    def _save_processed_url(self, url):
        """
        Adds a URL to the processed set and updates the JSON log to avoid duplicates.
        """
        self.processed_urls.add(url)
        with open(PROCESSED_LOG, "w") as f:
            json.dump(list(self.processed_urls), f)

    def _search_searxng(self, query, num_results=10):
        """
        Performs a SearxNG search against the configured SEARXNG_URL.
        Returns JSON results for the given query.
        """
        params = {
            "q": query,
            "format": "json",
            "engines": "google,bing,academic",
            "categories": "science",
            "safesearch": 1,
            "lang": "en",
            "count": num_results,
        }
        try:
            response = requests.get(urljoin(SEARXNG_URL, "/search"), params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"SearxNG search error for query '{query}': {e}")
            return {"results": []}  # Return empty results to avoid further errors

    def _extract_content_with_tika(self, url):
        """
        Fetches content from a URL and extracts text and metadata using Tika's /rmeta endpoint.
        Handles main document and attachments as separate parts.

        Returns a list of dicts (one dict per document part).
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            content_type = response.headers.get(
                "Content-Type", "application/octet-stream"
            )
            headers = {"Content-type": content_type, "Accept": "application/json"}
            tika_response = requests.put(
                f"{TIKA_URL}/rmeta", headers=headers, data=response.content
            )
            tika_response.raise_for_status()
            return json.loads(tika_response.text)
        except Exception as e:
            logging.error(f"Tika content extraction error for URL {url}: {e}")
            return []

    def _save_document_parts(self, url, content_parts, metadata):
        """
        Saves each document/attachment part as a JSON file in a subfolder named
        after the hash of the original URL.
        """
        try:
            url_hash = hashlib.md5(url.encode()).hexdigest()
            doc_folder = os.path.join(DOCUMENT_STORE, url_hash)
            os.makedirs(doc_folder, exist_ok=True)
            for i, part in enumerate(content_parts):
                filename = os.path.join(doc_folder, f"part_{i}.json")
                with open(filename, "w") as f:
                    json.dump(
                        {
                            "url": url,
                            "extraction_metadata": metadata,
                            "document_part": part,
                        },
                        f,
                        indent=2,
                    )
            logging.info(f"Saved {len(content_parts)} document part(s) for {url}.")
        except Exception as e:
            logging.error(f"Error saving document parts for {url}: {e}")

    def _crawl_links(self, start_url, depth=2):
        """
        Recursively crawls links within the same domain up to a specified depth,
        extracting and saving content from each encountered URL via Tika.
        """
        if start_url in self.processed_urls:
            logging.info(f"Skipping already processed URL: {start_url}")
            return

        try:
            response = requests.get(start_url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            content_parts = self._extract_content_with_tika(start_url)
            if content_parts:
                self._save_document_parts(start_url, content_parts, {"depth": depth})
                self._save_processed_url(start_url)

            if depth > 0:
                for link in soup.find_all("a", href=True):
                    absolute_url = urljoin(start_url, link["href"])
                    if urlparse(absolute_url).netloc == urlparse(start_url).netloc:
                        self._crawl_links(absolute_url, depth=depth - 1)

        except Exception as e:
            logging.error(f"Error crawling {start_url}: {e}")

    async def execute_search(self, query):
        """
        Executes a web search for the given query using SearxNG,
        crawls and extracts content from the search results, and saves the documents.

        Args:
            query (str): The search query.

        Returns:
            dict: JSON search results from SearxNG.
        """
        logging.info(f"Executing web search for query: {query}")
        search_results = await asyncio.to_thread(
            self._search_searxng, query
        )  # Run synchronous search in threadpool
        if search_results and search_results.get("results"):
            for result in search_results.get("results"):
                url = result.get("url")
                if url and url not in self.processed_urls:
                    logging.info(f"Processing URL from search result: {url}")
                    await asyncio.to_thread(
                        self._crawl_links, url
                    )  # Run synchronous crawl in threadpool
        return search_results


class ModelPool(ModelPoolInterface):
    """
    Concrete implementation of ModelPoolInterface, enhanced to load local models,
    fallback to Hugging Face, and demonstrate a potential self-improvement loop
    by allowing for different versions or fine-tuned models to be loaded.
    """

    def __init__(self):
        """
        Initializes the ModelPool with dictionaries to store models and track
        available model types.  Models are stored by their type (e.g., 'research', 'codegen').
        """
        self.models: Dict[str, Any] = (
            {}
        )  # Stores loaded model instances, keyed by model_type
        self.available_models: Deque[str] = deque()  # Queue of available model types
        self.optimizers: Dict[str, Any] = (
            {}
        )  # Stores optimizer model instances, keyed by optimizer_type
        self.available_optimizers: Deque[str] = (
            deque()
        )  # Queue of available optimizer types

        self.local_model_path = "/home/lloyd/Development/saved_models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.hf_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.checkpoint_dir = (
            "/home/lloyd/Development/offload"  # Add disk offload directory
        )
        self.local_optimizer_path = (
            "/home/lloyd/Development/saved_optimizers/adamw"  # Example path
        )
        self.hf_optimizer_name = "adamw_hf_name"  # Example name

    def get_adaptive_model(self, model_type: str = "qwen") -> Any:
        """
        Retrieves a model instance from the pool. If the model_type is not found
        or not yet loaded, it raises a ValueError.  This method could be extended
        to implement adaptive model selection logic based on task requirements
        and model performance metrics, contributing to a self-improvement loop.

        For now, it simply retrieves the model associated with the given model_type.
        """
        if model_type not in self.models:
            raise ValueError(
                f"Model type {model_type} not loaded. Available models are: {list(self.models.keys())}"
            )
        logging.info(f"Serving model of type: {model_type} from ModelPool.")
        return self.models[model_type]

    def get_optimizer(self, optimizer_type: str = "adamw") -> Any:
        """
        Retrieves an optimizer model instance from the pool.
        If the optimizer_type is not found or not yet loaded, it raises a ValueError.
        """
        if optimizer_type not in self.optimizers:
            raise ValueError(
                f"Optimizer type {optimizer_type} not loaded. Available optimizers are: {list(self.optimizers.keys())}"
            )
        logging.info(f"Serving optimizer of type: {optimizer_type} from ModelPool.")
        return self.optimizers[optimizer_type]

    def _load_model(self, model_type: str) -> Any:
        """
        Internal method to load a model from local path or Hugging Face.
        This is where the model loading and fallback logic resides.
        """
        local_path = self.local_model_path
        hf_name = self.hf_model_name

        try:
            # Load model with disk offloading
            model = AutoModelForCausalLM.from_pretrained(
                local_path if os.path.exists(local_path) else hf_name,
                trust_remote_code=True,
                device_map="auto",  # Enable automatic device mapping
            )

            # Apply disk offloading
            disk_offload(
                model=model,
                offload_dir=self.checkpoint_dir,
                execution_device=torch.device("cpu"),  # Use torch.device
                offload_buffers=True,
            )

            tokenizer = AutoTokenizer.from_pretrained(
                local_path if os.path.exists(local_path) else hf_name,
                trust_remote_code=True,
            )

            return {"model": model, "tokenizer": tokenizer}

        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

    def _load_optimizer(self, optimizer_type: str) -> Any:
        """
        Internal method to load an optimizer model.
        This is a placeholder as optimizers might be initialized differently.
        For now, it's a simplified loader, you might need to adjust based on
        how your optimizer models are structured and loaded.
        """
        local_optimizer_path = self.local_optimizer_path
        hf_optimizer_name = self.hf_optimizer_name

        try:
            # Placeholder for optimizer loading logic.
            # This might involve different loading mechanisms than language models.
            # For example, optimizers might be classes that need to be instantiated,
            # or configurations loaded from files.

            # Example: Assume optimizer is a class in a module.
            # module = importlib.import_module("my_optimizers") # hypothetical module
            # OptimizerClass = getattr(module, optimizer_type) # e.g., AdamWOptimizer
            # optimizer = OptimizerClass(**optimizer_config) # hypothetical config

            # For simplicity, let's return a string placeholder for now.
            optimizer = f"Optimizer_{optimizer_type}_loaded_successfully"
            logging.info(f"Optimizer type '{optimizer_type}' loaded (placeholder).")
            return optimizer

        except Exception as e:
            logging.error(f"Error loading optimizer {optimizer_type}: {e}")
            raise

    def preload_models(self, model_types: List[str]) -> None:
        """
        Preloads specified model types into the ModelPool. For each model type,
        it attempts to load the model using `_load_model` and stores it in the `self.models` dictionary.
        This simulates the pre-warming of the model pool, making models readily available.
        """
        logging.info(f"Preloading models: {model_types}")
        for model_type in model_types:
            if model_type not in self.models:  # Prevent reloading if already loaded
                try:
                    loaded_model = self._load_model(
                        model_type
                    )  # Load model using the _load_model method
                    if loaded_model:
                        self.models[model_type] = (
                            loaded_model  # Store the loaded model (dict containing model and tokenizer)
                        )
                        self.available_models.append(model_type)
                        logging.info(
                            f"Model type '{model_type}' preloaded successfully."
                        )
                    else:
                        logging.error(
                            f"Failed to load model type '{model_type}' during preload."
                        )
                except ValueError as e:
                    logging.error(f"Error preloading model type '{model_type}': {e}")
            else:
                logging.info(
                    f"Model type '{model_type}' already loaded. Skipping preload."
                )

    def preload_optimizers(self, optimizer_types: List[str]) -> None:
        """
        Preloads specified optimizer types into the ModelPool.
        """
        logging.info(f"Preloading optimizers: {optimizer_types}")
        for optimizer_type in optimizer_types:
            if optimizer_type not in self.optimizers:  # Prevent reloading
                try:
                    loaded_optimizer = self._load_optimizer(optimizer_type)
                    if loaded_optimizer:
                        self.optimizers[optimizer_type] = loaded_optimizer
                        self.available_optimizers.append(optimizer_type)
                        logging.info(
                            f"Optimizer type '{optimizer_type}' preloaded successfully."
                        )
                    else:
                        logging.error(
                            f"Failed to load optimizer type '{optimizer_type}' during preload."
                        )
                except Exception as e:
                    logging.error(
                        f"Error preloading optimizer type '{optimizer_type}': {e}"
                    )
            else:
                logging.info(
                    f"Optimizer type '{optimizer_type}' already loaded. Skipping preload."
                )

    def release_model(self, model_type: str) -> None:
        """
        Releases a model type back to the pool, conceptually making it available for reuse.
        In this basic implementation, it simply adds the model_type back to the available_models queue.
        In a more sophisticated system, this could involve resource management, model unloading, or other cleanup.
        """
        if model_type in self.models:
            if model_type not in self.available_models:  # Prevent adding duplicates
                self.available_models.append(model_type)
            logging.info(f"Model type '{model_type}' released back to pool.")
        else:
            logging.warning(f"Attempted to release unknown model type: {model_type}")

    def release_optimizer(self, optimizer_type: str) -> None:
        """
        Releases an optimizer type back to the pool.
        """
        if optimizer_type in self.optimizers:
            if optimizer_type not in self.available_optimizers:  # Prevent duplicates
                self.available_optimizers.append(optimizer_type)
            logging.info(f"Optimizer type '{optimizer_type}' released back to pool.")
        else:
            logging.warning(
                f"Attempted to release unknown optimizer type: {optimizer_type}"
            )

    def is_model_loaded(self, model_type: str) -> bool:
        """
        Checks if a model type is currently loaded in the pool.
        """
        return model_type in self.models

    def is_optimizer_loaded(self, optimizer_type: str) -> bool:
        """
        Checks if an optimizer type is currently loaded in the pool.
        """
        return optimizer_type in self.optimizers

    def get_available_model_types(self) -> List[str]:
        """
        Returns a list of currently available model types in the pool.
        """
        return list(self.models.keys())

    def get_available_optimizer_types(self) -> List[str]:
        """
        Returns a list of currently available optimizer types in the pool.
        """
        return list(self.optimizers.keys())


class QualityGateway:
    """Temporary quality gate implementation"""

    def validate_task(
        self, task: "Task"
    ) -> (
        bool
    ):  # Ensure type hint is string to avoid circular import if Task is in another file
        """
        Validates the incoming task. Currently, it's a placeholder that always returns True,
        effectively allowing all tasks to pass. In a real-world scenario, this would be replaced
        with actual quality checks, potentially using a smaller, faster model from the ModelPool
        to quickly assess task validity before routing to more resource-intensive submodules.
        """
        # Basic validation that always passes - replace with actual validation logic
        logging.debug(
            f"QualityGateway validating task: {task.task_id}. Currently always passing."
        )
        return True


def main():
    import asyncio
    import logging

    logging.basicConfig(level=logging.INFO)

    async def test_architecture():
        """Tests the modular local model network architecture with the updated ModelPool."""
        try:
            load_balancer = LoadBalancer()
            await load_balancer.initialize()  # Initialize load balancer and pre-warm resources
            workflow_coordinator = WorkflowCoordinator()

            # Preload the 'qwen' model type - this will attempt to load from local path, then HF
            model_types_to_preload = ["qwen", "research", "codegen", "analysis"]
            logging.info(f"Preloading initial models: {model_types_to_preload}")
            load_balancer.allocator.model_pool.preload_models(
                model_types_to_preload
            )  # preload qwen and other types
            logging.info("Initial model preloading complete.")

            # Example task payloads - one specifically requesting 'qwen' model type, others default
            task_payloads = [
                {
                    "task_id": "task_1",
                    "task_type": "research",
                    "content": "What is the capital of France?",
                },
                {
                    "task_id": "task_2",
                    "task_type": "codegen",
                    "instruction": "Write a python function to add two numbers.",
                },
                {
                    "task_id": "task_3",
                    "task_type": "analysis",
                    "data": "[1, 2, 3, 4, 5]",
                },
                {
                    "task_id": "task_4",
                    "task_type": "validation",
                    "output": "Paris is the capital.",
                },
                {
                    "task_id": "task_5",
                    "task_type": "optimization",
                    "model_name": "small_model",
                },
                {
                    "task_id": "task_6",
                    "task_type": "unknown",
                    "input": "some unknown task",
                },  # Test default routing
                {
                    "task_id": "task_7",
                    "task_type": "qwen_task",  # Example task type that might use the qwen model
                    "instruction": "Translate 'Hello world' to French using Qwen model.",
                },
            ]

            logging.info(f"Starting task processing for {len(task_payloads)} tasks.")
            for payload in task_payloads:
                logging.info(
                    f"Processing task: {payload['task_id']} with type: {payload.get('task_type', 'default')}"
                )
                try:
                    result = await workflow_coordinator.distribute_task(
                        payload, load_balancer
                    )
                    logging.info(f"Task {payload['task_id']} processed successfully.")
                    logging.debug(
                        f"Task {payload['task_id']} result: {result}"
                    )  # Log result in debug level

                except InvalidTaskError as e:
                    logging.warning(f"Task {payload['task_id']} failed validation: {e}")
                except Exception as e:
                    logging.error(
                        f"Error processing task {payload['task_id']}. Exception: {e}"
                    )
                    logging.exception(e)  # Log full exception traceback for debugging

            logging.info("All tasks processed in test_architecture.")

        except Exception as e:
            logging.critical(
                f"Critical error in test_architecture setup or execution: {e}"
            )
            logging.exception(e)  # Log full exception traceback for critical errors

    logging.info("Starting main test execution.")
    asyncio.run(test_architecture())
    logging.info("Main test execution completed.")


if __name__ == "__main__":
    main()
