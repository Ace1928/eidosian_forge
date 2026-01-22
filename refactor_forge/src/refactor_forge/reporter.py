"""
Reporter - Analysis Visualization Engine 📊

Renders analysis results in human-friendly formats with precise 
information layout and visual enhancements for clarity.
"""

from typing import Dict, Any, List, Optional
import networkx as nx
import textwrap


def print_analysis_report(results: Dict[str, Any]) -> None:
    """Print a human-readable report from analysis results.
    
    Args:
        results: Analysis results from analyze_code()
    """
    print(f"📊 Analysis Report for {results['file_info']['path']}")
    print(f"Found {len(results['modules'])} potential modules:")
    
    # Print modules information
    _print_modules_summary(results['modules'])
    
    # Print dependency graph
    if isinstance(results['dependencies'], nx.DiGraph) and results['dependencies'].number_of_edges() > 0:
        _print_dependency_graph(results['dependencies'])
    
    # Print import information
    _print_import_summary(results['symbols'])


def _print_modules_summary(modules: List[Dict[str, Any]]) -> None:
    """Print summary information for all detected modules.
    
    Args:
        modules: List of module dictionaries from analysis
    """
    for i, module in enumerate(modules, 1):
        print(f"\n{i}. {module['name']}")
        print(f"   Purpose: {module.get('purpose', 'Unknown purpose')}")
        print(f"   Lines: {module['start_line']+1}-{module['end_line']+1}")
        
        functions = module.get('functions', [])
        if functions:
            print(f"   Functions: {', '.join(f['name'] for f in functions)}")
        
        classes = module.get('classes', [])
        if classes:
            print(f"   Classes: {', '.join(c['name'] for c in classes)}")


def _print_dependency_graph(dependencies: nx.DiGraph) -> None:
    """Print a textual representation of the dependency graph.
    
    Args:
        dependencies: NetworkX DiGraph of module dependencies
    """
    print("\n⚡ Dependency Graph:")
    
    for module in dependencies.nodes():
        outgoing = list(dependencies.successors(module))
        if outgoing:
            print(f"   {module} → {', '.join(outgoing)}")
        else:
            print(f"   {module} (no dependencies)")


def _print_import_summary(symbols: Dict[str, Dict[str, Any]]) -> None:
    """Print summary of imported symbols.
    
    Args:
        symbols: Symbol table from analysis
    """
    if not symbols:
        return
        
    print("\n📦 Imports:")
    
    import_groups = {}
    for symbol, info in symbols.items():
        source = info.get('source', '')
        if source not in import_groups:
            import_groups[source] = []
        import_groups[source].append(symbol)
    
    for source, symbols_list in import_groups.items():
        if source:
            print(f"   From {source}: {', '.join(sorted(symbols_list))}")
        else:
            print(f"   Direct imports: {', '.join(sorted(symbols_list))}")
