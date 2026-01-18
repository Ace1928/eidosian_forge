"""
Analyzer Package - Code Structure Intelligence 🧠

Provides sophisticated tools for analyzing Python code structure,
detecting natural module boundaries, and identifying dependencies
between components.
"""

from pathlib import Path
from typing import Dict, Any, Union

from .code_analyzer import CodeAnalyzer

__all__ = ["analyze_code"]


def analyze_code(source_path: Union[str, Path]) -> Dict[str, Any]:
    """Analyze a Python source file for structural insights.
    
    High-level function that performs comprehensive code analysis
    and returns structured results.
    
    Args:
        source_path: Path to Python file to analyze
        
    Returns:
        Dict containing analysis results with modules, dependencies, etc.
        
    Raises:
        FileNotFoundError: If source file doesn't exist
    """
    analyzer = CodeAnalyzer(source_path)
    return analyzer.analyze()
