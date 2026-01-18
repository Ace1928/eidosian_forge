"""
Semantic Analyzer - Content Understanding 🧠

Extracts semantic meaning and purpose from code through analysis
of docstrings, comments, naming patterns, and structural context.
"""

import ast
import re
from typing import Dict, List, Any

from ..core.config import MODULE_PURPOSE_INDICATORS


def extract_semantic_purpose(modules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract semantic purpose and documentation from the code.
    
    Args:
        modules: List of identified modules
        
    Returns:
        Modules enhanced with semantic information
    """
    for module in modules:
        content = module["content"]
        
        # Extract functions and classes
        module_ast = ast.parse(content)
        module["functions"] = extract_functions(module_ast)
        module["classes"] = extract_classes(module_ast)
        
        # Try to extract purpose from docstrings
        docstring_match = re.search(r'"""(.*?)"""', content, re.DOTALL)
        if docstring_match:
            docstring = docstring_match.group(1).strip()
            # Get first line as purpose
            purpose = docstring.split('\n')[0].strip()
            module["purpose"] = purpose
            module["docstring"] = docstring
        else:
            # Try to extract from comments
            comment_match = re.search(r'# ([A-Za-z].*)', content)
            if comment_match:
                module["purpose"] = comment_match.group(1).strip()
            else:
                # Infer purpose from content patterns
                inferred_purpose = infer_purpose(content, module["name"])
                module["purpose"] = inferred_purpose
    
    return modules


def extract_functions(tree: ast.AST) -> List[Dict[str, Any]]:
    """Extract function information from the AST.
    
    Args:
        tree: AST to analyze
        
    Returns:
        List of function information dictionaries
    """
    functions = []
    
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append({
                "name": node.name,
                "lineno": node.lineno,
                "args": [arg.arg for arg in node.args.args],
                "docstring": ast.get_docstring(node) or ""
            })
    
    return functions


def extract_classes(tree: ast.AST) -> List[Dict[str, Any]]:
    """Extract class information from the AST.
    
    Args:
        tree: AST to analyze
        
    Returns:
        List of class information dictionaries
    """
    classes = []
    
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            methods = []
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    methods.append(child.name)
                    
            classes.append({
                "name": node.name,
                "lineno": node.lineno,
                "methods": methods,
                "docstring": ast.get_docstring(node) or ""
            })
    
    return classes


def infer_purpose(content: str, name: str) -> str:
    """Infer the purpose of a module from its content and name.
    
    Args:
        content: Module content
        name: Module name
        
    Returns:
        Inferred purpose string
    """
    # Check module name against known patterns
    for purpose, indicators in MODULE_PURPOSE_INDICATORS.items():
        for indicator in indicators:
            if indicator in name.lower():
                return f"{purpose.title()} module for {name.replace('_', ' ')}"
    
    # Check content for clues
    if "class" in content and "def __init__" in content:
        return f"Defines the {name.replace('_', ' ').title()} entity"
    elif content.count("def ") > 3:
        return f"Provides {name.replace('_', ' ')} functionality"
    
    return f"Handles {name.replace('_', ' ')} operations"
