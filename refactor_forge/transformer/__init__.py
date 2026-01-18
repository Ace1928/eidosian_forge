"""
Transformer Package - Code Generation Engine 🔧

Transforms analyzed code into perfectly modular architecture,
preserving functionality while enhancing structure.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union

from ..core.types import AnalysisResult, TransformationResult
from .code_generator import generate_package_structure
from .utils import resolve_output_path

__all__ = ["transform_code"]


def transform_code(
    analysis_results: Dict[str, Any],
    output_dir: Optional[Union[str, Path]] = None,
    package_name: Optional[str] = None
) -> Dict[str, Any]:
    """Transform analyzed code into a modular package structure.
    
    Args:
        analysis_results: Results from code analysis
        output_dir: Output directory (default: derived from source)
        package_name: Package name (default: derived from source)
        
    Returns:
        Transformation results with generated files
        
    Raises:
        ValueError: If analysis results are invalid
    """
    # Validate analysis results
    if not analysis_results or 'modules' not in analysis_results:
        raise ValueError("Invalid analysis results")
    
    # Resolve output path and package name
    source_path = Path(analysis_results['file_info']['path'])
    output_path, resolved_package = resolve_output_path(
        source_path, output_dir, package_name
    )
    
    # Generate package structure
    transformation = generate_package_structure(
        analysis_results, output_path, resolved_package
    )
    
    return transformation
