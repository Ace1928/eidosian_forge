#!/usr/bin/env python3
"""
🔮 EIDOSIAN SERIALIZATION DOCTOR v1.0

Ultra-precise diagnostic tool for identifying and repairing JSON serialization issues
in the Eidosian repository ecosystem. Implements quantum-level introspection with 
surgical precision healing capabilities.
"""
import sys
import json
import traceback
import importlib
import inspect
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# ANSI color codes for terminal output
BLUE = "\033[38;5;51m"
RED = "\033[38;5;196m"
GREEN = "\033[38;5;82m"
YELLOW = "\033[38;5;220m"
PURPLE = "\033[38;5;135m"
CYAN = "\033[38;5;39m"
RESET = "\033[0m"
BOLD = "\033[1m"

class SerializationInspector:
    """Hyper-precise diagnostic system for JSON serialization issues."""
    
    def __init__(self, log_path: Optional[Path] = None):
        """Initialize the inspection system with optional logging."""
        self.log_path = log_path
        self.issues_found = 0
        self.objects_inspected = 0
        self.repairs_made = 0
        
        if log_path:
            log_path.parent.mkdir(exist_ok=True, parents=True)
            with open(log_path, "w") as f:
                f.write(f"🔮 Eidosian Serialization Doctor Log - {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
    
    def log(self, message: str) -> None:
        """Record a message to the log file if enabled."""
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(f"{message}\n")
    
    def print_header(self, text: str) -> None:
        """Print a formatted section header."""
        print(f"\n{BLUE}{'═' * 70}{RESET}")
        print(f"{PURPLE}{BOLD}{text}{RESET}")
        print(f"{BLUE}{'═' * 70}{RESET}")
        self.log(f"\n{'=' * 70}\n{text}\n{'=' * 70}")
    
    def print_status(self, text: str, status: str = "info") -> None:
        """Print a status message with appropriate formatting."""
        color = {
            "info": CYAN,
            "success": GREEN,
            "error": RED,
            "warning": YELLOW
        }.get(status, CYAN)
        
        prefix = {
            "info": "ℹ️",
            "success": "✅",
            "error": "❌",
            "warning": "⚠️"
        }.get(status, "ℹ️")
        
        message = f"{prefix} {text}"
        print(f"{color}{message}{RESET}")
        self.log(message)
    
    def inspect_object(self, obj: Any, name: str = "root", path: str = "", depth: int = 0, 
                       max_depth: int = 10) -> Dict[str, Any]:
        """
        Perform atomic-level inspection of an object for serialization issues.
        
        Args:
            obj: The object to inspect
            name: Name of the current object
            path: JSON path to the current object
            depth: Current recursion depth
            max_depth: Maximum recursion depth
            
        Returns:
            Dictionary with inspection results
        """
        self.objects_inspected += 1
        current_path = f"{path}.{name}" if path else name
        
        # Prevent infinite recursion
        if depth > max_depth:
            return {"serializable": "unknown", "reason": "max depth exceeded", 
                   "path": current_path, "type": type(obj).__name__}
        
        # Check if object is directly serializable
        result = {
            "path": current_path,
            "type": type(obj).__name__,
            "serializable": True,
            "issues": []
        }
        
        # Try to serialize the object directly
        try:
            json.dumps(obj)
        except (TypeError, OverflowError) as e:
            result["serializable"] = False
            result["error"] = str(e)
            self.issues_found += 1
            
            # Perform deeper analysis based on type
            if isinstance(obj, dict):
                # Analyze each key-value pair
                problematic_keys = []
                for k, v in obj.items():
                    try:
                        json.dumps({k: v})
                    except (TypeError, OverflowError) as e:
                        problematic_keys.append({
                            "key": k,
                            "key_type": type(k).__name__,
                            "value_type": type(v).__name__,
                            "error": str(e)
                        })
                        
                        # Recursively inspect problematic values
                        if depth < max_depth - 1:
                            child_result = self.inspect_object(v, f"[{k!r}]", current_path, 
                                                              depth + 1, max_depth)
                            if not child_result.get("serializable", True):
                                result["issues"].append(child_result)
                
                result["problematic_keys"] = problematic_keys
                
            elif isinstance(obj, (list, tuple)):
                # Analyze each item in the sequence
                problematic_indices = []
                for i, item in enumerate(obj):
                    try:
                        json.dumps(item)
                    except (TypeError, OverflowError) as e:
                        problematic_indices.append({
                            "index": i,
                            "type": type(item).__name__,
                            "error": str(e)
                        })
                        
                        # Recursively inspect problematic items
                        if depth < max_depth - 1:
                            child_result = self.inspect_object(item, f"[{i}]", current_path, 
                                                             depth + 1, max_depth)
                            if not child_result.get("serializable", True):
                                result["issues"].append(child_result)
                
                result["problematic_indices"] = problematic_indices
            
            # For objects with __dict__ attribute, inspect their attributes
            elif hasattr(obj, "__dict__"):
                obj_dict = obj.__dict__
                child_result = self.inspect_object(obj_dict, "__dict__", current_path, 
                                                 depth + 1, max_depth)
                if not child_result.get("serializable", True):
                    result["issues"].append(child_result)
        
        return result
    
    def inspect_repo_forge(self) -> Dict[str, Any]:
        """Perform comprehensive inspection of repo_forge structures."""
        self.print_header("🔍 REPO_FORGE MODULE INSPECTION")
        
        try:
            import repo_forge
            from repo_forge.generators import scripts
            
            self.print_status(f"repo_forge version: {repo_forge.__version__}", "info")
            
            # Inspect create_script_files function returns
            self.print_status("Inspecting create_script_files return values...", "info")
            
            # Create minimal test environment
            test_path = Path("/tmp/eidosian_test")
            test_path.mkdir(exist_ok=True)
            
            # Call function with controlled environment
            try:
                result = scripts.create_script_files(
                    test_path, 
                    languages=["python"], 
                    overwrite=True
                )
                
                # Inspect the result
                inspection = self.inspect_object(result, "create_script_files_result")
                
                if not inspection.get("serializable", True):
                    self.print_status("❌ create_script_files return value is not serializable!", "error")
                    self.print_status(f"Issues detected: {len(inspection.get('issues', []))}", "warning")
                    
                    # Show problematic keys
                    if "problematic_keys" in inspection:
                        for item in inspection["problematic_keys"]:
                            self.print_status(
                                f"Problem with key {item['key']} ({item['key_type']}): {item['error']}", 
                                "warning"
                            )
                    
                    # Attempt repair
                    repaired = self._repair_result(result)
                    self.print_status("Attempting to repair result...", "info")
                    
                    # Check if repair was successful
                    try:
                        json.dumps(repaired)
                        self.print_status("✅ Repair successful!", "success")
                        self.repairs_made += 1
                        
                        # Show repair details
                        self.print_status(f"Original count: {result.get('count')} ({type(result.get('count')).__name__})", "info")
                        self.print_status(f"Repaired count: {repaired.get('count')} ({type(repaired.get('count')).__name__})", "info")
                        
                        # Show suggested patch
                        self._show_suggested_patch()
                        
                    except (TypeError, OverflowError) as e:
                        self.print_status(f"❌ Repair failed: {e}", "error")
                else:
                    self.print_status("✅ create_script_files return value is serializable!", "success")
            
            except Exception as e:
                self.print_status(f"❌ Error calling create_script_files: {e}", "error")
                traceback.print_exc()
            
            return {"success": True}
            
        except ImportError:
            self.print_status("❌ Could not import repo_forge module", "error")
            return {"success": False, "error": "import_error"}
    
    def _repair_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to repair a result dictionary for proper serialization."""
        if not isinstance(result, dict):
            return result
            
        repaired = {}
        
        # Process each key with type-specific handling
        for k, v in result.items():
            if k == "count" and isinstance(v, (int, float)):
                # Force to primitive int
                repaired[k] = int(v)
            elif k == "created_files" and isinstance(v, (list, tuple)):
                # Force each item to string
                repaired[k] = [str(item) for item in v]
            elif k == "base_path":
                # Force path to string
                repaired[k] = str(v)
            elif k == "_meta" and isinstance(v, dict):
                # Repair nested meta dictionary
                repaired[k] = self._repair_result(v)
            else:
                # Keep other values unchanged
                repaired[k] = v
                
        return repaired
    
    def _show_suggested_patch(self) -> None:
        """Show a suggested patch for the scripts.py file."""
        patch = """
# Add to repo_forge/core/diagnostics.py (or create if doesn't exist):
def ensure_serializable(obj):
    """Ensure an object is JSON serializable by converting problematic types."""
    if isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [ensure_serializable(i) for i in obj]
    elif isinstance(obj, (int, float, bool, str, type(None))):
        return obj
    else:
        return str(obj)

# Then in repo_forge/generators/scripts.py:
def create_script_files(...):
    # ... existing function ...
    
    # At the end of the function:
    result = {
        "success": True,
        "created_files": created_files,
        "count": int(len(created_files)),  # Explicit int cast
        "base_path": str(base_path),  # Explicit string cast
    }
    
    # Ensure JSON serializable (no _meta dictionary with complex types)
    from ..core.diagnostics import ensure_serializable
    return ensure_serializable(result)
"""
        self.print_status("Suggested fix:", "info")
        print(f"{CYAN}{patch}{RESET}")
        self.log(f"Suggested fix:\n{patch}")
    
    def run_diagnostics(self) -> None:
        """Run comprehensive diagnostics on the Eidosian system."""
        start_time = datetime.now()
        
        self.print_header("🔮 EIDOSIAN SERIALIZATION DOCTOR")
        self.print_status(f"Starting diagnostics at {start_time.isoformat()}", "info")
        
        # Inspect repo_forge
        repo_forge_results = self.inspect_repo_forge()
        
        # Test JSON encoders
        self.test_json_encoders()
        
        # Summarize results
        self.print_header("📊 DIAGNOSTIC SUMMARY")
        self.print_status(f"Objects inspected: {self.objects_inspected}", "info")
        self.print_status(f"Issues found: {self.issues_found}", "info" if self.issues_found == 0 else "warning")
        self.print_status(f"Repairs made: {self.repairs_made}", "info")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.print_status(f"Diagnostics completed in {duration:.2f} seconds", "success")
    
    def test_json_encoders(self) -> None:
        """Test and compare different JSON encoder strategies."""
        self.print_header("🧪 JSON ENCODER TESTING")
        
        test_obj = {
            "name": "test",
            "count": 42,
            "path": Path("/tmp"),
            "items": ["a", "b", "c"],
            "nested": {
                "key": "value",
                "number": 123.456
            }
        }
        
        # Test default encoder
        self.print_status("Testing default JSON encoder...", "info")
        try:
            result = json.dumps(test_obj)
            self.print_status("✅ Default encoder successful", "success")
        except (TypeError, OverflowError) as e:
            self.print_status(f"❌ Default encoder failed: {e}", "error")
            
            # Try custom encoder
            self.print_status("Testing custom JSON encoder...", "info")
            try:
                class CustomEncoder(json.JSONEncoder):
                    def default(self, obj):
                        if isinstance(obj, Path):
                            return str(obj)
                        return super().default(obj)
                
                result = json.dumps(test_obj, cls=CustomEncoder)
                self.print_status("✅ Custom encoder successful", "success")
            except (TypeError, OverflowError) as e:
                self.print_status(f"❌ Custom encoder failed: {e}", "error")
                
        # Test our repair strategy
        self.print_status("Testing repair strategy...", "info")
        repaired = self._repair_result(test_obj)
        try:
            result = json.dumps(repaired)
            self.print_status("✅ Repair strategy successful", "success")
        except (TypeError, OverflowError) as e:
            self.print_status(f"❌ Repair strategy failed: {e}", "error")


if __name__ == "__main__":
    # Initialize the serialization doctor with logging
    log_dir = Path.home() / "repos" / "eidosian_diagnostics"
    log_dir.mkdir(exist_ok=True, parents=True)
    log_path = log_dir / f"serialization_doctor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    doctor = SerializationInspector(log_path)
    doctor.run_diagnostics()