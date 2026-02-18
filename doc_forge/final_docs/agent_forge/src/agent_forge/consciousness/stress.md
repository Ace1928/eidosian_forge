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

This file contains the `stress.py` module, which is part of the agent forge project. It defines functions for managing consciousness stress benchmarks and emitting stress events.

## API Documentation

### Functions

#### _forge_root()
- **Purpose**: Returns the root directory where Eidos Forge data are stored.
- **Parameters**: None
- **Returns**:
  - A `Path` object representing the root directory of Eidos Forge data.

#### _stress_dir()
- **Purpose**: Returns a path to the stress benchmarks directory.
- **Parameters**: None
- **Returns**:
  - A `Path` object representing the stress benchmarks directory.

#### _now_iso()
- **Purpose**: Returns the current ISO formatted timestamp (e.g., "2023-01-29T14:35:00Z").
- **Parameters**: None
- **Returns**:
  - A `str` object representing the ISO formatted timestamp.

#### _load_json(path)
- **Purpose**: Loads a JSON file from the given path and returns its contents as a dictionary.
- **Parameters**:
  - `path`: Path to the JSON file.
- **Returns**:
  - An optional dictionary containing the loaded JSON data, or None if loading fails.

#### _safe_float(value, default=0.0)
- **Purpose**: Safely converts a value to a float and returns it with a fallback value if conversion is not possible (e.g., TypeError).
- **Parameters**:
  - `value`: The value to convert.
  - `default` (optional): The fallback value in case of failure.
- **Returns**:
  - A float or the default value.

#### _percentile(values, p)
- **Purpose**: Returns the percentile value from a list of values based on the provided percentiles.
- **Parameters**:
  - `values`: List of numeric values.
  - `p` (float): The desired percentile to return between 0 and 1.
- **Returns**:
  - A float representing the percentile value.

#### _event_bus_bytes(state_dir)
- **Purpose**: Returns the number of bytes in the event bus for the given state directory.
- **Parameters**:
  - `state_dir`: Path to the state directory.
- **Returns**:
  - An integer representing the total event bus size.

#### _iter_recent(state_dir, limit)
- **Purpose**: Iterates over recent events within a given state directory and returns them as a list of dictionaries.
- **Parameters**:
  - `state_dir`: Path to the state directory.
  - `limit` (int): The maximum number of items to return from the iterator.
- **Returns**:
  - A list of dictionaries containing recent event data.

#### _count_window(items, *, since_ts=None, etype="", source="")
- **Purpose**: Counts the occurrences of events within a specified window based on certain criteria.
- **Parameters**:
  - `items`: Iterable of event dictionaries.
  - `since_ts` (str): The minimum timestamp for the events to count.
  - `etype` (str): Event type filter, if any.
  - `source` (str): Source module filter, if any.
- **Returns**:
  - An integer representing the total count of events matching the criteria.

### Classes

#### _StressEmitterModule
- **Purpose**: Represents a stress emitter module for emitting synthetic and other types of events to manage consciousness stress.
- **Parameters**:
  - `event_fanout` (int): The number of fans out from this event source.
  - `broadcast_fanout` (int): The number of broadcasts from this event source.
  - `payload_chars` (int): The length in characters for payload strings.

#### StressBenchmarkResult
- **Purpose**: Represents a benchmark result object, containing information about the stress benchmark run.
- **Parameters**:
  - `benchmark_id`: A unique identifier for the benchmark.
  - `report_path`: Path to the JSON report file if persisted.
  - `report`: Report details as a dictionary.

#### ConsciousnessStressBenchmark
- **Purpose**: Represents a class for managing consciousness stress benchmarks and emitting events based on them.
- **Parameters**:
  - `state_dir`: A string or `Path` object representing the state directory.

### Methods

#### __init__(self, *, event_fanout: int, broadcast_fanout: int, payload_chars: int)
- **Purpose**: Initializes a new instance of the ConsciousnessStressBenchmark class.
- **Parameters**:
  - `event_fanout` (int): The number of fans out from this stress emitter module.
  - `broadcast_fanout` (int): The number of broadcasts from this stress emitter module.
  - `payload_chars` (int): The length in characters for payload strings.

#### latest_stress_benchmark(self)
- **Purpose**: Returns the latest stress benchmark result or None if there are no results found.
- **Parameters**:
  - `self.state_dir`: A string or `Path` object representing the state directory to find the latest stress benchmark.
- **Returns**:
  - An optional dictionary containing details about the latest stress benchmark, or None.

#### run(self, *, ticks: int = 20, event_fanout: int = 14, broadcast_fanout: int = 6, payload_chars: int = 12_000, max_payload_bytes: int = 2_048, persist: bool = True)
- **Purpose**: Runs a stress benchmark with specified parameters and returns the results as a `StressBenchmarkResult` object.
- **Parameters**:
  - `ticks`: Number of ticks to run for each event fanout/broadcast combination (default is 20).
  - `event_fanout`: Number of fans out from this event source (default is 14).
  - `broadcast_fanout`: Number of broadcasts from this event source (default is 6).
  - `payload_chars`: Length in characters for payload strings (default is 12_000).
  - `max_payload_bytes`: Maximum bytes allowed for payloads (default is 2_048).
  - `persist`: Whether to persist the results (default is True).

#### tick(self, ctx)
- **Purpose**: Updates the state directory and emits events based on certain criteria.
- **Parameters**:
  - `ctx`: A context object containing relevant information for emitting events.

#### iter_recent(self, state_dir: Path, limit=400)
- **Purpose**: Iterates over recent events within a given state directory and returns them as a list of dictionaries.

#### count_window(
    items: Iterable[dict[str, Any]],
    *,
    since_ts: str,
    etype: str = "",
    source: str = ""
)
- **Purpose**: Counts the occurrences of events within a

[ Prompt: 63.8 t/s | Generation: 8.0 t/s ]

Exiting...