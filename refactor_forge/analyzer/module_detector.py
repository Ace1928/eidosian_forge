"""
Module Detector - Boundary Intelligence 🧩

Identifies natural module boundaries in code using pattern recognition,
structural analysis, and semantic clustering techniques.
"""

import ast
import re
from typing import List, Dict, Any, Tuple

from ..core.config import COMPILED_BOUNDARY_PATTERNS


def detect_module_boundaries(source_code: str) -> List[Dict[str, Any]]:
    """Detect logical boundaries for module splitting.
    
    Args:
        source_code: Source code to analyze
        
    Returns:
        List of detected modules with their boundaries and content
    """
    lines = source_code.split('\n')
    tree = ast.parse(source_code)
    
    # Find sections based on boundary patterns
    section_boundaries = _find_section_boundaries(lines)
    
    # Find class and function definitions
    class_funcs = _find_definition_boundaries(tree)
    
    # Combine and sort all potential boundaries
    all_boundaries = sorted(set(section_boundaries + class_funcs))
    
    # Group closely related boundaries to form coherent modules
    grouped_boundaries = _group_related_boundaries(all_boundaries)
    
    # Convert grouped boundaries into module definitions
    return _create_module_definitions(grouped_boundaries, lines)


def _find_section_boundaries(lines: List[str]) -> List[int]:
    """Find section boundaries based on comment patterns.
    
    Args:
        lines: Source code lines
        
    Returns:
        Line numbers of section boundaries
    """
    boundaries = []
    for i, line in enumerate(lines):
        for pattern in COMPILED_BOUNDARY_PATTERNS:
            if pattern.match(line):
                boundaries.append(i)
    return boundaries


def _find_definition_boundaries(tree: ast.AST) -> List[int]:
    """Find class and function definition boundaries.
    
    Args:
        tree: AST of source code
        
    Returns:
        Line numbers of definition boundaries
    """
    class_funcs = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith('_'):
            # Check if this is a top-level definition
            if isinstance(ast.iter_fields(tree), ast.Module) or any(
                isinstance(parent, ast.Module) for parent in ast.iter_ancestors(node)
            ):
                class_funcs.append(node.lineno - 1)  # Convert to 0-based index
    return class_funcs


def _group_related_boundaries(boundaries: List[int], proximity_threshold: int = 10) -> List[List[int]]:
    """Group closely related boundaries together.
    
    Args:
        boundaries: Line numbers of potential boundaries
        proximity_threshold: Maximum lines between related boundaries
        
    Returns:
        Grouped boundaries forming coherent modules
    """
    grouped = []
    current_group = []
    
    for boundary in boundaries:
        if not current_group or boundary - current_group[-1] < proximity_threshold:
            current_group.append(boundary)
        else:
            grouped.append(current_group)
            current_group = [boundary]
    
    if current_group:
        grouped.append(current_group)
    
    return grouped


def _create_module_definitions(grouped_boundaries: List[List[int]], lines: List[str]) -> List[Dict[str, Any]]:
    """Create module definitions from grouped boundaries.
    
    Args:
        grouped_boundaries: Grouped line numbers forming modules
        lines: Source code lines
        
    Returns:
        List of module definitions with content and boundaries
    """
    modules = []
    
    for i, group in enumerate(grouped_boundaries):
        start_line = group[0]
        end_line = group[-1]
        
        # Expand to include associated code
        start_line = _find_module_start(start_line, lines)
        end_line = _find_module_end(end_line, lines)
        
        # Extract module name and code
        content = '\n'.join(lines[start_line:end_line + 1])
        name = _extract_module_name(content, f"module_{i}")
        
        modules.append({
            "name": name,
            "start_line": start_line,
            "end_line": end_line,
            "content": content
        })
    
    return modules


def _find_module_start(start_line: int, lines: List[str]) -> int:
    """Find the actual start of a module, considering preceding blank/comment lines.
    
    Args:
        start_line: Initial start line
        lines: Source code lines
        
    Returns:
        Adjusted start line
    """
    while start_line > 0:
        prev_line = lines[start_line - 1].strip()
        # Stop if we hit a boundary pattern or blank line after content
        if not prev_line or any(pattern.match(prev_line) for pattern in COMPILED_BOUNDARY_PATTERNS):
            break
        start_line -= 1
    return start_line


def _find_module_end(end_line: int, lines: List[str]) -> int:
    """Find the actual end of a module, considering trailing code.
    
    Args:
        end_line: Initial end line
        lines: Source code lines
        
    Returns:
        Adjusted end line
    """
    while end_line < len(lines) - 1:
        next_line = lines[end_line + 1].strip()
        # Stop if we hit a boundary pattern
        if any(pattern.match(next_line) for pattern in COMPILED_BOUNDARY_PATTERNS):
            break
        if not next_line and end_line + 2 < len(lines) and not lines[end_line + 2].strip():
            # Two blank lines often indicate a section break
            break
        end_line += 1
    return end_line


def _extract_module_name(content: str, default: str) -> str:
    """Extract appropriate module name from content.
    
    Args:
        content: Module content
        default: Default name if extraction fails
        
    Returns:
        Extracted module name
    """
    # Try to find a class name
    class_match = re.search(r"class ([A-Z][a-zA-Z0-9_]*)", content)
    if class_match:
        return class_match.group(1).lower()
    
    # Look for section headers
    header_match = re.search(r"# [│╭╰][ ]*([A-Za-z ]+?)[ ]*[│╮╯]", content)
    if header_match:
        # Convert "Section Name" to "section_name"
        section_name = header_match.group(1).strip().lower().replace(' ', '_')
        return section_name
    
    # Look for main function
    func_match = re.search(r"def ([a-z][a-zA-Z0-9_]*)\(", content)
    if func_match:
        return func_match.group(1)
    
    return default
