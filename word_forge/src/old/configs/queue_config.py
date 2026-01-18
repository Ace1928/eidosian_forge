"""
Queue configuration for Word Forge asynchronous processing.

This module defines the configuration schema for word processing queues,
including performance parameters, throttling, threading, and metrics
collection for optimizing asynchronous word processing operations.

Architecture:
    ┌───────────────┐
    │  QueueConfig  │
    └───────┬───────┘
            │
    ┌───────┴───────┐
    │   Components  │
    └───────────────┘
    ┌─────┬─────┬─────┬─────┬─────┐
    │Batch│Throt│Locks│Metr │Cache│
    └─────┴─────┴─────┴─────┴─────┘
"""

from dataclasses import dataclass
from functools import cached_property
from typing import ClassVar, Dict, Optional, Union

from word_forge.configs.config_essentials import (
    LockType,
    QueueMetricsFormat,
    QueuePerformanceProfile,
)
from word_forge.configs.config_types import EnvMapping


@dataclass
class QueueConfig:
    """
    Configuration for word queue processing.

    Controls batch sizes, throttling, threading, locking mechanisms, and
    performance parameters for the asynchronous word processing queue.

    Attributes:
        batch_size: Default batch size for processing operations
        throttle_seconds: Processing throttle duration in seconds
        lru_cache_size: Cache size for seen items lookup optimization
        max_queue_size: Maximum queue size (None = unlimited)
        apply_default_normalization: Whether to apply default normalization
        use_threading: Whether to use threading for queue processing
        lock_type: Type of lock to use ("reentrant" or "standard")
        track_metrics: Whether to track performance metrics
        metrics_format: Format for storing/exporting metrics data
        max_sample_size: Maximum sample size for metrics collection
        performance_profile: Preset performance profile to apply

    Usage:
        from word_forge.config import config

        # Access settings
        batch_size = config.queue.batch_size

        # Apply performance profile
        optimized_config = config.queue.with_performance_profile(
            QueuePerformanceProfile.HIGH_THROUGHPUT
        )
    """

    # Processing batch settings
    batch_size: int = 50
    throttle_seconds: float = 0.1

    # Cache and queue limits
    lru_cache_size: int = 128
    max_queue_size: Optional[int] = None  # None = unlimited

    # Processing options
    apply_default_normalization: bool = True

    # Thread safety settings
    use_threading: bool = True
    lock_type: LockType = "reentrant"

    # Performance monitoring
    track_metrics: bool = False
    metrics_format: QueueMetricsFormat = "json"
    max_sample_size: int = 100

    # Performance profiles
    performance_profile: Optional[QueuePerformanceProfile] = None

    # Environment variable mapping for configuration overrides
    ENV_VARS: ClassVar[EnvMapping] = {
        "WORD_FORGE_QUEUE_BATCH_SIZE": ("batch_size", int),
        "WORD_FORGE_QUEUE_THROTTLE": ("throttle_seconds", float),
        "WORD_FORGE_QUEUE_CACHE_SIZE": ("lru_cache_size", int),
        "WORD_FORGE_QUEUE_MAX_SIZE": ("max_queue_size", int),
        "WORD_FORGE_QUEUE_USE_THREADS": ("use_threading", bool),
        "WORD_FORGE_QUEUE_TRACK_METRICS": ("track_metrics", bool),
        "WORD_FORGE_QUEUE_PERFORMANCE_PROFILE": (
            "performance_profile",
            QueuePerformanceProfile,
        ),
    }

    @cached_property
    def effective_throughput(self) -> float:
        """
        Calculate effective throughput in items per second.

        Returns:
            float: Theoretical maximum items processed per second
        """
        if self.throttle_seconds <= 0:
            return float(self.batch_size)  # Avoid division by zero
        return self.batch_size / self.throttle_seconds

    def with_batch_size(self, batch_size: int) -> "QueueConfig":
        """
        Create a new config instance with modified batch size.

        Args:
            batch_size: New batch size value

        Returns:
            QueueConfig: New instance with updated batch size
        """
        return QueueConfig(
            batch_size=batch_size,
            throttle_seconds=self.throttle_seconds,
            lru_cache_size=self.lru_cache_size,
            max_queue_size=self.max_queue_size,
            apply_default_normalization=self.apply_default_normalization,
            use_threading=self.use_threading,
            lock_type=self.lock_type,
            track_metrics=self.track_metrics,
            metrics_format=self.metrics_format,
            max_sample_size=self.max_sample_size,
            performance_profile=self.performance_profile,
        )

    def with_performance_profile(
        self, profile: QueuePerformanceProfile
    ) -> "QueueConfig":
        """
        Apply a predefined performance profile to the configuration.

        Args:
            profile: Performance profile to apply

        Returns:
            QueueConfig: New instance with profile settings applied
        """
        # Start with current config
        config = QueueConfig(
            batch_size=self.batch_size,
            throttle_seconds=self.throttle_seconds,
            lru_cache_size=self.lru_cache_size,
            max_queue_size=self.max_queue_size,
            apply_default_normalization=self.apply_default_normalization,
            use_threading=self.use_threading,
            lock_type=self.lock_type,
            track_metrics=self.track_metrics,
            metrics_format=self.metrics_format,
            max_sample_size=self.max_sample_size,
            performance_profile=profile,
        )

        # Apply profile-specific settings
        if profile == QueuePerformanceProfile.LOW_LATENCY:
            config.batch_size = 10
            config.throttle_seconds = 0.01
            config.lru_cache_size = 256
        elif profile == QueuePerformanceProfile.HIGH_THROUGHPUT:
            config.batch_size = 200
            config.throttle_seconds = 0.5
            config.lru_cache_size = 512
        elif profile == QueuePerformanceProfile.MEMORY_EFFICIENT:
            config.batch_size = 25
            config.throttle_seconds = 0.3
            config.lru_cache_size = 64
        # BALANCED profile uses the default values

        return config

    def optimize_for_threading(self, enable: bool) -> "QueueConfig":
        """
        Optimize configuration for threaded or non-threaded operation.

        Args:
            enable: Whether to enable threading optimizations

        Returns:
            QueueConfig: New instance with threading optimizations
        """
        config = QueueConfig(
            batch_size=self.batch_size,
            throttle_seconds=self.throttle_seconds,
            lru_cache_size=self.lru_cache_size,
            max_queue_size=self.max_queue_size,
            apply_default_normalization=self.apply_default_normalization,
            use_threading=enable,
            lock_type=self.lock_type,
            track_metrics=self.track_metrics,
            metrics_format=self.metrics_format,
            max_sample_size=self.max_sample_size,
            performance_profile=self.performance_profile,
        )

        # Apply threading-specific optimizations
        if enable:
            config.lock_type = "reentrant"
        else:
            # No locking needed without threading
            config.batch_size = min(
                config.batch_size * 2, 500
            )  # Can process more at once

        return config

    def get_metrics_config(self) -> Dict[str, Union[bool, str, int]]:
        """
        Get configuration subset for metrics collection.

        Returns:
            Dict[str, Union[bool, str, int]]: Metrics-specific configuration
        """
        return {
            "enabled": self.track_metrics,
            "format": self.metrics_format,
            "max_sample_size": self.max_sample_size,
        }


# ==========================================
# Module Exports
# ==========================================

__all__ = [
    "QueueConfig",
    "LockType",
    "QueuePerformanceProfile",
    "QueueMetricsFormat",
]
