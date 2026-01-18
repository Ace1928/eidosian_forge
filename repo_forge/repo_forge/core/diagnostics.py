"""
Eidosian Surgical Diagnostics System.

Provides ultra-precise error detection and data integrity verification
with atomic-level granularity for perfect debugging.
"""
import sys
import json
import logging
import traceback
from typing import Any, Dict, Optional, Union
from datetime import datetime
from pathlib import Path

# Integration with Eidosian Foundation
try:
    from gis_forge import GisCore
    from diagnostics_forge import DiagnosticsForge
    HAS_FOUNDATION = True
except ImportError:
    HAS_FOUNDATION = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚡ Core Diagnostic Functions - Quantum-Level Precision
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DiagnosticLogger:
    """Hyper-detailed logging system with context tracing."""
    
    def __init__(self, name: str = "repo_forge.diagnostics", log_path: Optional[Path] = None):
        """Initialize the advanced diagnostic logger."""
        self.gis = GisCore() if HAS_FOUNDATION else None
        
        # Use DiagnosticsForge if available for underlying implementation
        if HAS_FOUNDATION:
            log_dir = str(log_path.parent) if log_path else "logs"
            self.forge_diag = DiagnosticsForge(log_dir=log_dir, service_name=name)
            self.logger = self.forge_diag.logger
        else:
            self.logger = logging.getLogger(name)
            self.logger.setLevel(logging.DEBUG)
            self._setup_standard_logging(log_path)
            self.forge_diag = None

    def _setup_standard_logging(self, log_path: Optional[Path]):
        # Maintain default console handler for immediate visibility
        if not self.logger.handlers:
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)
            console_format = logging.Formatter(
                "🔍 %(levelname)s [%(asctime)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s",
                datefmt="%H:%M:%S"
            )
            console.setFormatter(console_format)
            self.logger.addHandler(console)
            
            # Add file handler if path provided
            if log_path:
                try:
                    file_handler = logging.FileHandler(log_path, mode='a')
                    file_handler.setLevel(logging.DEBUG)
                    file_format = logging.Formatter(
                        "%(levelname)s [%(asctime)s] %(name)s.%(funcName)s:%(lineno)d - %(message)s"
                    )
                    file_handler.setFormatter(file_format)
                    self.logger.addHandler(file_handler)
                except Exception as e:
                    self.logger.warning(f"Failed to create log file at {log_path}: {e}")
    
    def trace(self, message: str, data: Any = None, stack_level: int = 2) -> None:
        """Log with ultra-precise stack trace and data inspection."""
        frame = sys._getframe(stack_level)
        filename = frame.f_code.co_filename
        lineno = frame.f_lineno
        function = frame.f_code.co_name
        
        # Generate execution context
        context = f"{Path(filename).name}:{lineno} in {function}()"
        
        if data is not None:
            # Special handling for different data types with introspection
            data_type = type(data).__name__
            data_details = {}
            
            # Perform deep introspection based on type
            if hasattr(data, "__dict__"):
                data_details = {
                    "attributes": str(data.__dict__),
                    "methods": [m for m in dir(data) if callable(getattr(data, m)) and not m.startswith("__")]
                }
            
            # For basic types, include value-specific details
            if isinstance(data, (int, float, bool, str)):
                data_details["value"] = str(data)
                if isinstance(data, int):
                    data_details["hex"] = hex(data)
                    data_details["bin"] = bin(data)
            elif isinstance(data, dict):
                data_details["keys"] = list(data.keys())
                data_details["key_types"] = {k: type(k).__name__ for k in data.keys()}
                data_details["value_types"] = {k: type(v).__name__ for k, v in data.items()}
            
            try:
                json_str = json.dumps(data)
                data_details["json_serializable"] = True
                if len(json_str) > 500:
                    data_details["json_sample"] = json_str[:500] + "..."
            except (TypeError, ValueError, OverflowError) as e:
                data_details["json_serializable"] = False
                data_details["json_error"] = str(e)
            
            log_msg = f"[{context}] {message} | Data({data_type}): {data_details}"
        else:
            log_msg = f"[{context}] {message}"
            
        self.logger.debug(log_msg)

    def start_timer(self, name: str):
        if self.forge_diag:
            return self.forge_diag.start_timer(name)
        return {"name": name, "start": datetime.now()}

    def stop_timer(self, timer: Dict[str, Any]):
        if self.forge_diag:
            return self.forge_diag.stop_timer(timer)
        duration = (datetime.now() - timer["start"]).total_seconds()
        self.logger.debug(f"Timer {timer['name']} stopped: {duration}s")
        return duration


# Create globally accessible instance with standard configuration
diagnostics = DiagnosticLogger()

def verify_json_serializable(data: Any, key: str = "root") -> None:
    """
    Verify an object is JSON serializable with detailed diagnostics.
    
    Args:
        data: Data to verify
        key: Path/key for the data (for nested structures)
    
    Raises:
        ValueError with detailed diagnostic information on failure
    """
    try:
        json.dumps(data)
        diagnostics.trace(f"✅ Verified JSON serializable: {key}", data)
    except Exception as e:
        # Generate comprehensive diagnostic report
        report = diagnostics.capture_serialization(data)
        diagnostics.logger.error(f"❌ JSON serialization failed for {key}: {e}")
        diagnostics.logger.error(f"Diagnostic report: {json.dumps(report, indent=2)}")
        raise ValueError(f"JSON serialization error in {key}: {str(e)}") from e

def safe_int(value: Any) -> int:
    """
    Convert value to int with exhaustive validation and diagnostics.
    
    Args:
        value: Value to convert to int
        
    Returns:
        Integer value with guaranteed primitive int type
    """
    # Type-specific conversion with detailed diagnostics
    original_type = type(value).__name__
    
    try:
        # Force primitive int conversion
        result = int(value)
        
        # Verify no data was lost in conversion
        if isinstance(value, (int, float)) and float(result) != float(value):
            diagnostics.logger.warning(
                f"⚠️ Potential data loss in int conversion: {value} ({original_type}) -> {result}"
            )
        
        diagnostics.trace(
            f"Int conversion: {value} ({original_type}) -> {result} ({type(result).__name__})",
            {"original": value, "result": result}
        )
        return result
    except (TypeError, ValueError) as e:
        diagnostics.logger.error(f"❌ Int conversion failed for {value} ({original_type}): {e}")
        # Fall back to zero with an error log
        return 0

def sanitize_for_json(obj: Any) -> Any:
    """
    Aggressively sanitize any object for JSON serialization.
    
    Args:
        obj: Any Python object
        
    Returns:
        JSON-serializable version with strictly primitive types
    """
    # Handle dictionaries - most common source of issues
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    
    # Handle lists and tuples
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    
    # Force primitive types for common values
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, int):
        return int(obj)
    elif isinstance(obj, float):
        return float(obj)
    elif isinstance(obj, str):
        return str(obj)
    elif obj is None:
        return None
    
    # Convert everything else to strings
    else:
        return str(obj)