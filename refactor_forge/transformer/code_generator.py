"""
Code Generator - Module Assembly 🏗️

Generates optimized, modular code structure from analysis results,
creating a perfectly organized package.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

from ..core.types import TransformationFile, TransformationResult
from ..core.config import (
    MODULE_DOCSTRING_TEMPLATE, 
    PACKAGE_INIT_TEMPLATE,
    README_TEMPLATE,
    DEFAULT_MODULE_EMOJIS
)


def generate_package_structure(
    analysis_results: Dict[str, Any], 
    output_path: Path,
    package_name: str
) -> TransformationResult:
    """Generate a complete package structure from analysis results.
    
    Args:
        analysis_results: Results from code analysis
        output_path: Path where package will be created
        package_name: Name of the package
        
    Returns:
        Transformation results with generated files
    """
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize the list of files to generate
    files: List[TransformationFile] = []
    module_map: Dict[str, str] = {}
    
    # Create package level __init__.py
    init_file = _generate_package_init(analysis_results, package_name)
    files.append({
        "path": str(output_path / "__init__.py"),
        "content": init_file
    })
    
    # Create README
    readme = _generate_readme(analysis_results, package_name)
    files.append({
        "path": str(output_path / "README.md"),
        "content": readme
    })
    
    # Create module files
    for module in analysis_results['modules']:
        module_file, module_path = _generate_module_file(
            module, package_name, output_path
        )
        
        files.append({
            "path": str(module_path),
            "content": module_file
        })
        
        module_map[module['name']] = str(module_path.relative_to(output_path))
    
    return {
        "output_path": str(output_path),
        "files": files,
        "package_name": package_name,
        "module_map": module_map
    }


def _generate_package_init(analysis_results: Dict[str, Any], package_name: str) -> str:
    """Generate the package __init__.py file.
    
    Args:
        analysis_results: Results from code analysis
        package_name: Name of the package
        
    Returns:
        Content for __init__.py file
    """
    source_file = analysis_results['file_info']['path']
    package_description = f"Modular version of {os.path.basename(source_file)}"
    
    # Determine exports based on module content
    exports = []
    for module in analysis_results['modules']:
        # For each module that has functions or classes, we'll export them
        if module.get('functions') or module.get('classes'):
            mod_name = module['name']
            exports.append(f"from .{mod_name} import *")
    
    imports_str = "\n".join(exports)
    
    original_source = f"Generated from {os.path.basename(source_file)}"
    
    return PACKAGE_INIT_TEMPLATE.format(
        package_name=package_name,
        package_description=package_description,
        original_source_comment=original_source,
        imports=imports_str
    )


def _generate_readme(analysis_results: Dict[str, Any], package_name: str) -> str:
    """Generate README.md for the package.
    
    Args:
        analysis_results: Results from code analysis
        package_name: Name of the package
        
    Returns:
        Content for README.md
    """
    source_file = analysis_results['file_info']['path']
    package_description = f"Modular version of {os.path.basename(source_file)}"
    
    # Generate module list
    module_list = ""
    for module in analysis_results['modules']:
        purpose = module.get('purpose', 'Unknown purpose')
        module_list += f"- **{module['name']}**: {purpose}\n"
    
    # Find a good example import
    example_import = next(
        (m['name'] for m in analysis_results['modules'] 
         if m.get('functions') or m.get('classes')),
        analysis_results['modules'][0]['name'] if analysis_results['modules'] else "module"
    )
    
    return README_TEMPLATE.format(
        package_name=package_name,
        package_description=package_description,
        module_list=module_list,
        package_import_name=package_name,
        example_import=example_import,
        source_file=os.path.basename(source_file)
    )


def _generate_module_file(
    module: Dict[str, Any], 
    package_name: str,
    output_path: Path
) -> Tuple[str, Path]:
    """Generate an individual module file.
    
    Args:
        module: Module information from analysis
        package_name: Name of the package
        output_path: Base output path
        
    Returns:
        Tuple of (file content, file path)
    """
    module_name = module['name']
    module_path = output_path / f"{module_name}.py"
    
    # Extract or infer module purpose and emoji
    purpose = module.get('purpose', 'Module')
    
    # Determine appropriate emoji
    emoji = DEFAULT_MODULE_EMOJIS["default"]
    for key, value in DEFAULT_MODULE_EMOJIS.items():
        if key in module_name.lower() or key in purpose.lower():
            emoji = value
            break
    
    # Extract docstring from original content or create one
    if "docstring" in module:
        docstring = module["docstring"]
    else:
        description = f"Functionality extracted from {package_name}"
        docstring = MODULE_DOCSTRING_TEMPLATE.format(
            module_name=module_name.replace('_', ' ').title(),
            module_purpose=purpose,
            emoji=emoji,
            module_description=description
        )
    
    # Extract module content, removing the original docstring if present
    content = module['content']
    content = _remove_existing_docstring(content)
    
    # Add standardized docstring and imports
    full_content = f'{docstring}\n\n{content}'
    
    return full_content, module_path


def _remove_existing_docstring(content: str) -> str:
    """Remove existing docstring from code content.
    
    Args:
        content: Source code content
        
    Returns:
        Content with docstring removed
    """
    import ast
    
    try:
        # Parse the content
        tree = ast.parse(content)
        
        # Check if there's a module docstring
        if (tree.body and isinstance(tree.body[0], ast.Expr) and 
                isinstance(tree.body[0].value, ast.Str)):
            # Get the end line of the docstring
            end_line = tree.body[0].end_lineno
            
            # Skip the lines containing the docstring
            lines = content.split('\n')
            return '\n'.join(lines[end_line:])
        
        return content
    except SyntaxError:
        # If parsing fails, return the original content
        return content
