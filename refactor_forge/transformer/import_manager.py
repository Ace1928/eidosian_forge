"""
Import Manager - Dependency Orchestration 📦

Analyzes and restructures import relationships between modules to ensure
clean dependencies and proper symbol resolution in transformed code.
"""

import ast
import re
from typing import Dict, List, Set, Any, Tuple


def reorganize_imports(
    modules: List[Dict[str, Any]], 
    dependency_graph: Any
) -> Dict[str, List[str]]:
    """Reorganize imports between modules based on dependency graph.
    
    Args:
        modules: List of module information
        dependency_graph: NetworkX dependency graph
        
    Returns:
        Dictionary mapping module names to required imports
    """
    module_imports = {}
    
    # For each module, determine required imports
    for module in modules:
        module_name = module["name"]
        required_imports = []
        
        # Add direct dependencies
        try:
            dependencies = list(dependency_graph.successors(module_name))
            for dep in dependencies:
                required_imports.append(dep)
        except (KeyError, TypeError):
            # Module might not be in graph
            pass
            
        module_imports[module_name] = required_imports
    
    # Check for circular dependencies and resolve
    module_imports = _resolve_circular_dependencies(module_imports)
    
    return module_imports


def _resolve_circular_dependencies(
    module_imports: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """Resolve circular dependencies in import structure.
    
    Args:
        module_imports: Dictionary of module imports
        
    Returns:
        Cleaned import dictionary
    """
    # Build dependency matrix
    modules = list(module_imports.keys())
    n = len(modules)
    
    # Initialize adjacency matrix
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    
    # Fill matrix with direct dependencies
    module_indices = {name: i for i, name in enumerate(modules)}
    
    for module, imports in module_imports.items():
        i = module_indices[module]
        for imp in imports:
            if imp in module_indices:
                j = module_indices[imp]
                matrix[i][j] = 1
    
    # Find cycles
    visited = [0] * n
    recursion_stack = [0] * n
    cycles = []
    
    def dfs(node, path):
        visited[node] = 1
        recursion_stack[node] = 1
        
        for neighbor in range(n):
            if matrix[node][neighbor] == 1:
                if recursion_stack[neighbor] == 1:
                    # Cycle found
                    cycle = path + [modules[neighbor]]
                    if cycle not in cycles:
                        cycles.append(cycle)
                elif visited[neighbor] == 0:
                    dfs(neighbor, path + [modules[neighbor]])
        
        recursion_stack[node] = 0
    
    for i in range(n):
        if visited[i] == 0:
            dfs(i, [modules[i]])
    
    # Break cycles by removing the weakest dependency in each cycle
    for cycle in cycles:
        # Find the dependency with the fewest imports
        min_imports = float('inf')
        weakest_link = (cycle[0], cycle[1])
        
        for i in range(len(cycle)):
            from_module = cycle[i]
            to_module = cycle[(i+1) % len(cycle)]
            
            count = len(module_imports.get(from_module, []))
            if count < min_imports:
                min_imports = count
                weakest_link = (from_module, to_module)
        
        # Remove the weakest dependency
        if weakest_link[1] in module_imports.get(weakest_link[0], []):
            module_imports[weakest_link[0]].remove(weakest_link[1])
    
    return module_imports


def extract_used_symbols(content: str) -> Set[str]:
    """Extract symbols that are actually used in the code content.
    
    Args:
        content: Source code content
        
    Returns:
        Set of symbol names used in code
    """
    try:
        tree = ast.parse(content)
        symbols = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                symbols.add(node.id)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                symbols.add(node.func.id)
                
        return symbols
    except SyntaxError:
        # Fallback to simple regex for partial code fragments
        pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
        return set(re.findall(pattern, content))


def optimize_imports(
    source: str, 
    module_imports: Dict[str, List[str]]
) -> Dict[str, List[str]]:
    """Optimize imports by removing unused dependencies.
    
    Args:
        source: Original source code
        module_imports: Original module import mapping
        
    Returns:
        Optimized module import mapping
    """
    # For each module, check which imports are actually used
    optimized = {}
    
    for module, imports in module_imports.items():
        # Find the module content in source modules
        try:
            module_tree = ast.parse(source)
            module_symbols = set()
            
            # Extract used symbols
            for node in ast.walk(module_tree):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    module_symbols.add(node.id)
            
            # Keep only imports with used symbols
            optimized[module] = [imp for imp in imports if any(
                symbol in module_symbols for symbol in extract_used_symbols(imp)
            )]
        except SyntaxError:
            # If parsing fails, keep all imports
            optimized[module] = imports
    
    return optimized
