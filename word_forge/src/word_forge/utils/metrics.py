from __future__ import annotations

import logging
from typing import Dict, Optional

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from eidosian_core import eidosian

LOGGER = logging.getLogger("word_forge.utils.metrics")

# Metrics definitions
WORDS_PROCESSED = None
PROCESSING_ERRORS = None
QUEUE_SIZE = None
WORKER_STATE = None
PROCESSING_LATENCY = None

if PROMETHEUS_AVAILABLE:
    WORDS_PROCESSED = Counter(
        "word_forge_words_processed_total",
        "Total number of words processed by the refiner"
    )
    PROCESSING_ERRORS = Counter(
        "word_forge_processing_errors_total",
        "Total number of errors encountered during word processing"
    )
    QUEUE_SIZE = Gauge(
        "word_forge_queue_size",
        "Number of items currently in the processing queue"
    )
    WORKER_STATE = Gauge(
        "word_forge_worker_status",
        "Current state of background workers",
        ["worker_name", "state"]
    )
    PROCESSING_LATENCY = Histogram(
        "word_forge_processing_duration_seconds",
        "Time spent processing a single word",
        buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, float("inf"))
    )

@eidosian()
def initialize_metrics(port: int = 8000) -> bool:
    """Start the Prometheus metrics server."""
    if not PROMETHEUS_AVAILABLE:
        LOGGER.warning("Prometheus client not available, metrics disabled")
        return False
    
    try:
        start_http_server(port)
        LOGGER.info(f"Prometheus metrics server started on port {port}")
        return True
    except Exception as e:
        LOGGER.error(f"Failed to start Prometheus server: {e}")
        return False

@eidosian()
def record_word_processed() -> None:
    """Increment the processed words counter."""
    if WORDS_PROCESSED:
        WORDS_PROCESSED.inc()

@eidosian()
def record_error() -> None:
    """Increment the processing errors counter."""
    if PROCESSING_ERRORS:
        PROCESSING_ERRORS.inc()

@eidosian()
def update_queue_size(size: int) -> None:
    """Update the queue size gauge."""
    if QUEUE_SIZE:
        QUEUE_SIZE.set(size)

@eidosian()
def set_worker_state(name: str, state: str, value: float) -> None:
    """Set the worker state gauge."""
    if WORKER_STATE:
        WORKER_STATE.labels(worker_name=name, state=state).set(value)

@eidosian()
def observe_latency(duration: float) -> None:
    """Record processing latency in the histogram."""
    if PROCESSING_LATENCY:
        PROCESSING_LATENCY.observe(duration)
