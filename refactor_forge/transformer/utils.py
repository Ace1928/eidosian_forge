"""
Transformer Utilities - Support Functions 🛠️

Provides utility functions for the transformation process, including
path resolution, import management, and code formatting.
"""

import re
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List, Any

from ..core.utils import derive_package_name, derive_output_dir


def resolve_output_path(
    source_path: Path,
    output_dir: Optional[Union[str, Path]] = None,
    package_name: Optional[str] = None
) -> Tuple[Path, str]:
    """Resolve the output path and package name.
    
    Args:
        source_path: Path to source file
        output_dir: Optional output directory
        package_name: Optional package name
        
    Returns:
        Tuple of (output path, package name)
    """
    # Determine package name
    resolved_package = package_name or derive_package_name(source_path)
    
    # Determine output directory
    if output_dir:
        output_path = Path(output_dir)
    else:
        output_path = derive_output_dir(source_path, resolved_package)
    
    return output_path, resolved_package


def sanitize_imports(
    content: str, 
    original_package: str, 
    new_package: str
) -> str:
    """Update import statements to reflect new package structure.
    
    Args:
        content: Source code
        original_package: Original package/module name
        new_package: New package name
        
    Returns:
        Updated source code
    """
    # Handle direct imports
    pattern = rf"import\s+{re.escape(original_package)}(\s+as\s+\w+)?"
    replacement = f"import {new_package}\\1"
    content = re.sub(pattern, replacement, content)
    
    # Handle from imports
    pattern = rf"from\s+{re.escape(original_package)}\s+import"
    replacement = f"from {new_package} import"
    content = re.sub(pattern, replacement, content)
    
    return content


def format_code(content: str, max_line_length: int = 88) -> str:
    """Format code to ensure consistent style.
    
    Args:
        content: Source code to format
        max_line_length: Maximum line length for wrapping
        
    Returns:
        Formatted code
    """
    try:
        # Try using Black if available
        import black
        mode = black.Mode(line_length=max_line_length)
        formatted = black.format_str(content, mode=mode)
        return formatted
    except (ImportError, ModuleNotFoundError):
        # Basic formatting as fallback
        lines = content.splitlines()
        formatted_lines = []
        
        # Simple line handling
        for line in lines:
            # Keep line breaks
            if not line.strip():
                formatted_lines.append("")
                continue
                
            # Handle import grouping
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                formatted_lines.append(line)
                continue
            
            # Preserve indentation
            indent_match = re.match(r"^(\s+)", line)
            indent = indent_match.group(1) if indent_match else ""
            
            # Simple wrap for long lines
            if len(line) > max_line_length:
                # For function calls and parameters
                if "(" in line and ")" in line:
                    parts = re.split(r"([(,)])", line)
                    current_line = ""
                    for part in parts:
                        if len(current_line + part) > max_line_length and current_line.strip():
                            formatted_lines.append(current_line)
                            current_line = indent + "    " + part.lstrip()
                        else:
                            current_line += part
                    if current_line.strip():
                        formatted_lines.append(current_line)
                else:
                    formatted_lines.append(line)
            else:
                formatted_lines.append(line)
        
        return "\n".join(formatted_lines)


def generate_imports(module_info: Dict[str, Any], dependencies: List[str]) -> str:
    """Generate import statements for a module based on its dependencies.
    
    Args:
        module_info: Module information from analysis
        dependencies: List of module dependencies
        
    Returns:
        Import statements as string
    """
    imports = []
    
    # Standard library imports
    std_lib_imports = set()
    for func in module_info.get("functions", []):
        # Add common imports based on function usage patterns
        if "path" in func["name"].lower() or "file" in func["name"].lower():
            std_lib_imports.add("from pathlib import Path")
        if "json" in func["name"].lower():
            std_lib_imports.add("import json")
        if any(arg.endswith("dict") for arg in func["args"]):
            std_lib_imports.add("from typing import Dict")
    
    # Type imports
    type_imports = set()
    if module_info.get("functions") or module_info.get("classes"):
        type_imports.add("from typing import Dict, List, Optional, Any, Union")
    
    # Dependency imports
    dep_imports = set()
    for dep in dependencies:
        dep_imports.add(f"from .{dep} import *")
    
    # Add imports in correct order
    if std_lib_imports:
        imports.extend(sorted(std_lib_imports))
        imports.append("")
    
    if type_imports:
        imports.extend(sorted(type_imports))
        imports.append("")
    
    if dep_imports:
        imports.extend(sorted(dep_imports))
        imports.append("")
    
    return "\n".join(imports)
