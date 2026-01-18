"""
Dependency Analyzer - Relationship Mapping 🔗

Maps dependencies between code components by analyzing
symbol usage and reference patterns across modules.
"""

import ast
from typing import Dict, List, Set, Any

import networkx as nx


def build_dependency_graph(modules: List[Dict[str, Any]]) -> nx.DiGraph:
    """Build a directed graph of dependencies between modules.
    
    Args:
        modules: List of module dictionaries
        
    Returns:
        NetworkX DiGraph representing module dependencies
    """
    dependencies = nx.DiGraph()
    
    # Add nodes for all modules
    for module in modules:
        dependencies.add_node(module["name"], info=module)
    
    # Add dependency edges
    for i, module in enumerate(modules):
        module_ast = ast.parse(module["content"])
        
        # Extract all names referenced in this module
        referenced_names = extract_referenced_names(module_ast)
        
        # Check if names are defined in other modules
        for j, other_module in enumerate(modules):
            if i == j:
                continue
                
            other_ast = ast.parse(other_module["content"])
            defined_names = extract_defined_names(other_ast)
            
            # If this module references names defined in the other module, add dependency
            if referenced_names.intersection(defined_names):
                dependencies.add_edge(module["name"], other_module["name"])
    
    return dependencies


def extract_referenced_names(tree: ast.AST) -> Set[str]:
    """Extract all names that are referenced (used) in the AST.
    
    Args:
        tree: AST to analyze
        
    Returns:
        Set of referenced name strings
    """
    referenced_names = set()
    
    for node in ast.walk(tree):
        # Variables being accessed
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            referenced_names.add(node.id)
        
        # Function calls
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            referenced_names.add(node.func.id)
            
        # Attribute access (obj.attr)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if isinstance(node.value, ast.Name):
                referenced_names.add(node.value.id)
    
    return referenced_names


def extract_defined_names(tree: ast.AST) -> Set[str]:
    """Extract all names that are defined in the AST.
    
    Args:
        tree: AST to analyze
        
    Returns:
        Set of defined name strings
    """
    defined_names = set()
    
    for node in ast.walk(tree):
        # Function definitions
        if isinstance(node, ast.FunctionDef):
            defined_names.add(node.name)
            
        # Class definitions
        elif isinstance(node, ast.ClassDef):
            defined_names.add(node.name)
            
        # Variable assignments
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined_names.add(target.id)
                    
        # Variable annotations
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            defined_names.add(node.target.id)
    
    return defined_names
