"""
Import Analyzer - Symbol Detection 📚

Parses and analyzes import statements to build a comprehensive
symbol table and track external dependencies.
"""

import ast
from typing import Dict, Any


def analyze_imports(source_code: str) -> Dict[str, Dict[str, Any]]:
    """Analyze imports in the source code and build symbol table.
    
    Args:
        source_code: Source code to analyze
        
    Returns:
        Symbol table with import information
    """
    tree = ast.parse(source_code)
    symbol_table = {}
    
    for node in ast.walk(tree):
        # Direct imports: import module, import module as alias
        if isinstance(node, ast.Import):
            for name in node.names:
                symbol_table[name.asname or name.name] = {
                    "type": "import",
                    "source": name.name,
                    "alias": name.asname,
                    "lineno": node.lineno
                }
        
        # From imports: from module import name, from module import name as alias
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            for name in node.names:
                symbol_table[name.asname or name.name] = {
                    "type": "import_from",
                    "source": module,
                    "name": name.name,
                    "alias": name.asname,
                    "lineno": node.lineno
                }
    
    return symbol_table


def categorize_imports(symbol_table: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Categorize imports by type (standard lib, third-party, local).
    
    Args:
        symbol_table: Symbol table from analyze_imports
        
    Returns:
        Categorized imports dictionary
    """
    standard_lib = {}
    third_party = {}
    local = {}
    
    import sys
    import pkgutil
    
    # Get list of standard library modules
    stdlib_modules = set(m[1] for m in pkgutil.iter_modules(sys.stdlib_module_names))
    
    for symbol, info in symbol_table.items():
        source = info["source"] if info["type"] == "import" else info.get("source", "")
        top_level = source.split('.')[0]
        
        if top_level in stdlib_modules:
            standard_lib[symbol] = info
        elif source.startswith('.'):
            local[symbol] = info
        else:
            third_party[symbol] = info
    
    return {
        "standard_lib": standard_lib,
        "third_party": third_party,
        "local": local
    }
