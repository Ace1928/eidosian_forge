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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ⚡ Core Diagnostic Functions - Quantum-Level Precision
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class DiagnosticLogger:
    """Hyper-detailed logging system with context tracing."""
    
    def __init__(self, name: str = "eidosian.diagnostics", log_path: Optional[Path] = None):
        """Initialize the advanced diagnostic logger."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Maintain default console handler for immediate visibility
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
    
    def trace(self, message: str, data: Any = None, stack_level: int = 1) -> None:
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
                # For numeric types, show binary/hex representation for complete clarity
                if isinstance(data, int):
                    data_details["hex"] = hex(data)
                    data_details["bin"] = bin(data)
            elif isinstance(data, dict):
                # For dictionaries, analyze keys and values
                data_details["keys"] = list(data.keys())
                data_details["key_types"] = {k: type(k).__name__ for k in data.keys()}
                data_details["value_types"] = {k: type(v).__name__ for k, v in data.items()}
            
            # Try safe JSON serialization with error handling
            try:
                json_str = json.dumps(data)
                data_details["json_serializable"] = True
                # Only include sample for large data
                if len(json_str) > 500:
                    data_details["json_sample"] = json_str[:500] + "..."
            except (TypeError, ValueError, OverflowError) as e:
                data_details["json_serializable"] = False
                data_details["json_error"] = str(e)
                # Try to identify problematic keys
                if isinstance(data, dict):
                    problem_keys = []
                    for k, v in data.items():
                        try:
                            json.dumps({k: v})
                        except Exception as e:
                            problem_keys.append((k, type(v).__name__, str(e)))
                    if problem_keys:
                        data_details["problematic_fields"] = problem_keys
            
            log_msg = f"[{context}] {message} | Data({data_type}): {data_details}"
        else:
            log_msg = f"[{context}] {message}"
            
        self.logger.debug(log_msg)

    def capture_serialization(self, data: Any) -> Dict[str, Any]:
        """
        Analyze data for JSON serialization issues with surgical precision.
        
        Returns a comprehensive diagnostic report with specific problem areas identified.
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "serializable": False,
            "issues": [],
            "type_report": {},
        }
        
        # Build type report - recursive introspection
        report["type_report"] = self._build_type_report(data)
        
        # Attempt serialization with surgical error trapping
        try:
            json_str = json.dumps(data)
            report["serializable"] = True
            report["string_length"] = len(json_str)
            report["sample"] = json_str[:100] + "..." if len(json_str) > 100 else json_str
            return report
        except (TypeError, ValueError, OverflowError) as e:
            report["error"] = str(e)
            report["error_type"] = type(e).__name__
            
            # For dictionaries, perform atomic key-value pair analysis
            if isinstance(data, dict):
                for k, v in data.items():
                    try:
                        # Test each key-value pair individually
                        json.dumps({k: v})
                    except Exception as pair_error:
                        # Record precise details about problematic pairs
                        report["issues"].append({
                            "key": str(k),
                            "key_type": type(k).__name__,
                            "value_type": type(v).__name__,
                            "error": str(pair_error),
                            "value_sample": str(v)[:50] if isinstance(v, str) else str(v),
                        })
            elif isinstance(data, (list, tuple)):
                # For sequences, identify problematic indices
                for i, item in enumerate(data):
                    try:
                        json.dumps(item)
                    except Exception as item_error:
                        report["issues"].append({
                            "index": i,
                            "type": type(item).__name__,
                            "error": str(item_error),
                            "value_sample": str(item)[:50] if isinstance(item, str) else str(item),
                        })
        
        return report
    
    def _build_type_report(self, data: Any, depth: int = 0, max_depth: int = 3) -> Dict[str, Any]:
        """Build recursive type report with configurable depth."""
        if depth > max_depth:
            return {"type": type(data).__name__, "truncated": True}
        
        if isinstance(data, dict):
            return {
                "type": "dict",
                "length": len(data),
                "keys": {
                    k: self._build_type_report(v, depth + 1, max_depth)
                    for k, v in data.items()
                } if depth < max_depth else "truncated"
            }
        elif isinstance(data, (list, tuple)):
            return {
                "type": type(data).__name__,
                "length": len(data),
                "items": [
                    self._build_type_report(item, depth + 1, max_depth) 
                    for item in data[:5]  # Limit items for large collections
                ] if depth < max_depth else "truncated",
                "truncated": len(data) > 5
            }
        else:
            # Base case for primitive types
            type_info = {"type": type(data).__name__}
            
            # Add extra details for numeric types
            if isinstance(data, int):
                type_info["value"] = data
                type_info["hex"] = hex(data)
            elif isinstance(data, (float, bool, str)):
                type_info["value"] = data
                
            return type_info


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