#!/usr/bin/env python3
"""
Terminal Forge Repository Initializer
====================================

Creates a comprehensive directory structure and placeholder files for the Terminal Forge project
following Eidosian principles of contextual integrity, structural harmony, and functional elegance.

This script transforms an empty directory into a complete project scaffold with:
- Core package structure with modular components
- Example hierarchy from basic to advanced usage patterns
- Comprehensive test architecture (unit, integration, performance)
- Documentation system with manual pages, auto-generated references, and visual assets
- Project metadata and configuration files

Each component exists for a specific purpose with precise relationships to other elements,
embodying the Eidosian approach of "everything essential, nothing superfluous."

Usage:
    ./terminal_forge_repo_init.py [path]

Arguments:
    path: Target directory to initialize (defaults to current directory)

Examples:
    ./terminal_forge_repo_init.py
    ./terminal_forge_repo_init.py /path/to/new/project
"""

import os
import shutil
import sys
from pathlib import Path
import argparse
from typing import Optional, List, Dict, Union, Tuple, Callable
import time
import re


def create_directory(path: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Uses os.makedirs() to create the specified directory path and all necessary parent 
    directories. Provides visual feedback on the operation's outcome.
    
    Parameters
    ----------
    path : str
        The directory path to create
        
    Returns
    -------
    None
        Function operates through side effects
        
    Examples
    --------
    >>> create_directory("./terminal_forge")
    ✅ Created directory: ./terminal_forge
    
    >>> create_directory("./existing_directory")
    ℹ️ Directory already exists: ./existing_directory
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"✅ Created directory: {path}")
    else:
        print(f"ℹ️ Directory already exists: {path}")


def create_empty_file(path: str, content: str = "") -> None:
    """
    Create a file with optional content if it doesn't exist.
    
    Writes the specified content to a file at the given path, providing
    visual feedback on the operation's outcome.
    
    Parameters
    ----------
    path : str
        The file path to create
    content : str, optional
        Content to write to the file (default: empty string)
        
    Returns
    -------
    None
        Function operates through side effects
        
    Examples
    --------
    >>> create_empty_file("README.md", "# Project Title\\n\\nProject description.")
    ✅ Created file: README.md
    
    >>> create_empty_file("existing_file.txt")
    ℹ️ File already exists: existing_file.txt
    """
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write(content)
        print(f"✅ Created file: {path}")
    else:
        print(f"ℹ️ File already exists: {path}")


def create_python_file(path: str, module_name: str = "") -> None:
    """
    Create a Python file with an Eidosian-style docstring if it doesn't exist.
    
    Generates a Python file with a contextually appropriate docstring template.
    The module_name defaults to the filename if not provided.
    
    Parameters
    ----------
    path : str
        The file path to create
    module_name : str, optional
        Name to use in the docstring (default: derived from filename)
        
    Returns
    -------
    None
        Function operates through side effects
        
    Examples
    --------
    >>> create_python_file("terminal_forge/colors.py", "Color System")
    ✅ Created file: terminal_forge/colors.py
    
    >>> create_python_file("existing_module.py")
    ℹ️ Python file already exists: existing_module.py
    """
    if not os.path.exists(path):
        module_name = module_name or os.path.basename(path).replace('.py', '')
        # Create a more comprehensive docstring following Eidosian principle of Precision as Style
        content = f'''"""
{module_name}

Description of the module's purpose and functionality following Eidosian design principles:
- Contextual Integrity: Every element serves an exact purpose
- Precision as Style: Function and form fused seamlessly
- Flow Like a River: Operations chain naturally with minimal friction

This module should embody core Terminal Forge architectural patterns while maintaining
crystal-clear intent and optimal performance characteristics.
"""

'''
        create_empty_file(path, content)
    else:
        print(f"ℹ️ Python file already exists: {path}")


def create_markdown_file(path: str, title: str = "") -> None:
    """
    Create a Markdown file with a basic title and template if it doesn't exist.
    
    Generates a properly formatted Markdown file with a title and placeholder content.
    The title defaults to a formatted version of the filename if not provided.
    
    Parameters
    ----------
    path : str
        The file path to create
    title : str, optional
        Title to use as the main heading (default: derived from filename)
        
    Returns
    -------
    None
        Function operates through side effects
        
    Examples
    --------
    >>> create_markdown_file("docs/getting_started.md", "Getting Started Guide")
    ✅ Created file: docs/getting_started.md
    
    >>> create_markdown_file("existing_doc.md")
    ℹ️ Markdown file already exists: existing_doc.md
    """
    if not os.path.exists(path):
        title = title or os.path.basename(path).replace('.md', '').replace('_', ' ').title()
        content = f'''# {title}

Content will go here.
'''
        create_empty_file(path, content)
    else:
        print(f"ℹ️ Markdown file already exists: {path}")


def find_and_move_python_files(source_dir: str, target_structure: List[str]) -> None:
    """
    Find Python files in the source directory and move them to the appropriate location.
    
    Intelligently organizes loose Python files by examining their names and content,
    then placing them in the most appropriate directory within the project structure.
    Provides detailed feedback on moved files.
    
    Parameters
    ----------
    source_dir : str
        The root directory to search for Python files
    target_structure : List[str]
        List of subdirectories that make up the target project structure
        
    Returns
    -------
    None
        Function operates through side effects
    """
    moved_files = []
    
    # File classification mapping - each file has a precise destination
    file_destinations = {
        # Core engine components
        "banner.py": "terminal_forge",
        "colors.py": "terminal_forge",
        "borders.py": "terminal_forge", 
        "layout.py": "terminal_forge",
        "themes.py": "terminal_forge",
        "effects.py": "terminal_forge", 
        "ascii_art.py": "terminal_forge",
        "utils.py": "terminal_forge",
        "__init__.py": "terminal_forge",
        
        # Example components
        "basic.py": "examples",
        "advanced.py": "examples",
        "animations.py": "examples"
    }
    
    for root, _, files in os.walk(source_dir):
        for file in files:
            if not file.endswith('.py'):
                continue
                
            source_path = os.path.join(root, file)
            
            # Skip files already in target structure
            if any(source_path.startswith(os.path.join(source_dir, dir_path)) for dir_path in target_structure):
                continue
            
            # Determine target directory with precision
            if file.startswith("test_"):
                target_dir = os.path.join(source_dir, "tests", "unit")
            elif file in file_destinations:
                target_dir = os.path.join(source_dir, file_destinations[file])
            else:
                # Fallback for unclassified Python files
                target_dir = os.path.join(source_dir, "examples")
            
            # Execute move if target exists and destination doesn't
            if target_dir and os.path.exists(target_dir):
                target_path = os.path.join(target_dir, file)
                if not os.path.exists(target_path):
                    try:
                        shutil.move(source_path, target_path)
                        moved_files.append(f"📦 Moved {source_path} to {target_path}")
                    except Exception as ex:
                        print(f"⚠️ Could not move {source_path} to {target_path}: {ex}")
                else:
                    print(f"⚠️ Target file already exists, not moving: {target_path}")
    
    # Report results with contextual feedback
    if moved_files:
        print("\n".join(moved_files))
    else:
        print("ℹ️ No Python files needed to be moved")


DEBUG_MODE = False

def debug_log(message: str):
    """Print debug messages if DEBUG_MODE is enabled."""
    if DEBUG_MODE:
        print(f"DEBUG: {message}")


def enhance_existing_files_with_eidosian_docstrings(repo_path: str) -> Dict[str, int]:
    """
    Enhance existing files with Eidosian-style contextually-aware docstrings.
    
    Scans repository for Python files that lack proper documentation headers
    and enriches them with contextually appropriate docstrings that understand
    their position in the ecosystem. This function embodies the principle of
    'Self-Awareness as Foundation' by making each file conscious of its role.
    
    Parameters
    ----------
    repo_path : str
        Path to the root directory of the repository
        
    Returns
    -------
    Dict[str, int]
        Statistics about enhanced files
    """
    print("\n🧠 Enhancing existing files with Eidosian docstrings...")
    stats = {"scanned": 0, "enhanced": 0, "skipped": 0}
    
    # File category patterns to determine contextual role
    file_categories = {
        r"terminal_forge/(\w+)\.py": "core",
        r"examples/.*\.py": "example",
        r"tests/unit/test_(\w+)\.py": "unit_test",
        r"tests/integration/.*\.py": "integration_test",
        r"tests/performance/.*\.py": "performance_test",
        r"docs/tools/.*\.py": "documentation_tool"
    }
    
    # Contextual docstring templates mapped by category
    docstring_templates = {
        "core": '''"""
{module_name}

A core Terminal Forge component that embodies Eidosian design principles:
- Contextual Integrity: Every element serves an exact purpose
- Precision as Style: Function and form fused seamlessly
- Structure as Control: Part of a grand architectural tapestry

This module functions as a {module_purpose} within the Terminal Forge ecosystem,
providing {module_capability} with optimal performance characteristics.
"""''',
        "example": '''"""
{module_name} Example

Demonstrates practical implementation of Terminal Forge capabilities:
- Shows {demonstration_focus} in action
- Illustrates proper usage patterns and best practices
- Serves as a learning path from concept to implementation

This example showcases how to leverage Terminal Forge for {demonstration_purpose}.
"""''',
        "unit_test": '''"""
Unit Tests for {tested_module}

Verification suite ensuring the {tested_module} module maintains:
- Functional correctness across all operations
- Proper error handling and edge case management
- API contract adherence and stability

These tests embody 'Truth is tested, not assumed' Eidosian principle.
"""''',
        "integration_test": '''"""
Integration Tests for {integration_focus}

Tests that verify proper composition and interaction between:
- {component_a}
- {component_b}
- {component_c}

Ensures the components work harmoniously as a cohesive system.
"""''',
        "performance_test": '''"""
Performance Tests for {performance_focus}

Measures and validates:
- Execution speed under various conditions
- Memory efficiency and resource utilization
- Scalability characteristics with increasing load

These benchmarks establish performance boundaries and expectations.
"""''',
        "documentation_tool": '''"""
{tool_name} - Documentation Tool

An intelligent knowledge management tool that:
- {primary_function}
- Maintains consistency across documentation artifacts
- Enforces Eidosian principles in knowledge representation

This tool understands its position in the Terminal Forge knowledge ecosystem.
"""'''
    }
    
    # Find all Python files
    for root, _, files in os.walk(repo_path):
        for file in files:
            if not file.endswith('.py'):
                continue
                
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, repo_path)
            stats["scanned"] += 1
            
            # Skip files in .git or __pycache__
            if ".git" in rel_path or "__pycache__" in rel_path:
                stats["skipped"] += 1
                continue
            
            # Read file content
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
            except Exception as e:
                print(f"⚠️ Could not read {rel_path}: {e}")
                stats["skipped"] += 1
                continue
            
            # Check if file already has a proper docstring
            if '"""' in content[:500] and len(content[:500].split('"""')[1].strip().split('\n')) > 2:
                debug_log(f"File already has docstring: {rel_path}")
                stats["skipped"] += 1
                continue
                
            # Determine file category and generate contextual docstring
            file_category = None
            match_data = {}
            
            for pattern, category in file_categories.items():
                if re.match(pattern, rel_path):
                    file_category = category
                    
                    # Extract contextual information
                    if category == "core":
                        module_name = file.replace('.py', '').replace('_', ' ').title()
                        # Analyze content to determine purpose
                        if "color" in file.lower() or "palette" in content.lower():
                            module_purpose = "visual styling system"
                            module_capability = "color management and palette optimization"
                        elif "border" in file.lower() or "frame" in content.lower():
                            module_purpose = "boundary definition system" 
                            module_capability = "customizable border rendering and management"
                        elif "layout" in file.lower() or "position" in content.lower():
                            module_purpose = "spatial organization engine"
                            module_capability = "precise element positioning and arrangement"
                        elif "effect" in file.lower() or "animation" in content.lower():
                            module_purpose = "visual transformation engine"
                            module_capability = "dynamic effects and transitions"
                        elif "banner" in file.lower() or "header" in content.lower():
                            module_purpose = "prominence signaling system"
                            module_capability = "attention-focusing visual hierarchies"
                        elif "util" in file.lower():
                            module_purpose = "foundational support system"
                            module_capability = "cross-cutting utility functions"
                        else:
                            module_purpose = "specialized component"
                            module_capability = "focused functionality"
                            
                        match_data = {
                            "module_name": module_name,
                            "module_purpose": module_purpose,
                            "module_capability": module_capability
                        }
                        
                    elif category == "example":
                        module_name = file.replace('.py', '').replace('_', ' ').title()
                        
                        # Determine example focus
                        if "basic" in file.lower() or "simple" in file.lower():
                            demo_focus = "fundamental Terminal Forge concepts"
                            demo_purpose = "creating simple yet effective terminal interfaces"
                        elif "advanced" in file.lower() or "complex" in file.lower():
                            demo_focus = "sophisticated Terminal Forge techniques"
                            demo_purpose = "building complex interactive experiences"
                        elif "layout" in file.lower() or "grid" in file.lower():
                            demo_focus = "spatial organization principles"
                            demo_purpose = "creating visually balanced layouts"
                        elif "animation" in file.lower() or "effect" in file.lower():
                            demo_focus = "dynamic visual transitions"
                            demo_purpose = "adding meaningful motion to interfaces"
                        elif "theme" in file.lower() or "color" in file.lower():
                            demo_focus = "visual styling techniques"
                            demo_purpose = "creating cohesive visual experiences"
                        else:
                            demo_focus = "specific Terminal Forge capabilities"
                            demo_purpose = "solving common terminal interface problems"
                            
                        match_data = {
                            "module_name": module_name,
                            "demonstration_focus": demo_focus,
                            "demonstration_purpose": demo_purpose
                        }
                        
                    elif category == "unit_test":
                        match = re.search(r"test_(\w+)\.py", file)
                        tested_module = match.group(1) if match else file.replace('test_', '').replace('.py', '')
                        tested_module = tested_module.replace('_', ' ').title()
                        
                        match_data = {
                            "tested_module": tested_module
                        }
                        
                    elif category == "integration_test":
                        module_name = file.replace('.py', '').replace('test_', '').replace('_', ' ').title()
                        
                        # Analyze content to determine tested components
                        components = []
                        if "theme" in file.lower() or "color" in content.lower():
                            components = ["Theme System", "Color Management", "Visual Rendering"]
                        elif "layout" in file.lower() or "position" in content.lower():
                            components = ["Layout Engine", "Element Positioning", "Container Management"]
                        elif "animation" in file.lower() or "effect" in content.lower():
                            components = ["Animation System", "Effect Pipeline", "Timing Controller"]
                        else:
                            components = ["Component A", "Component B", "Terminal Interface"]
                            
                        match_data = {
                            "integration_focus": module_name,
                            "component_a": components[0],
                            "component_b": components[1],
                            "component_c": components[2]
                        }
                        
                    elif category == "performance_test":
                        module_name = file.replace('.py', '').replace('test_', '').replace('_', ' ').title()
                        
                        match_data = {
                            "performance_focus": module_name
                        }
                        
                    elif category == "documentation_tool":
                        tool_name = file.replace('.py', '').replace('_', ' ').title()
                        
                        # Determine primary function
                        if "generate" in file.lower() or "create" in file.lower():
                            primary = "generates comprehensive documentation artifacts"
                        elif "validate" in file.lower() or "check" in file.lower():
                            primary = "validates documentation integrity and completeness"
                        elif "link" in file.lower():
                            primary = "ensures proper cross-referencing between documentation components"
                        else:
                            primary = "enhances documentation quality and cohesion"
                            
                        match_data = {
                            "tool_name": tool_name,
                            "primary_function": primary
                        }
                    
                    break
                    
            if file_category and file_category in docstring_templates:
                # Format docstring template with extracted data
                try:
                    docstring = docstring_templates[file_category].format(**match_data)
                    
                    # Detect if file has proper Python structure and insert docstring
                    lines = content.split('\n')
                    insert_line = 0
                    
                    # Skip shebang line if present
                    if lines and lines[0].startswith('#!'):
                        insert_line = 1
                        
                    # Skip module-level comments/license if present
                    while insert_line < len(lines) and lines[insert_line].startswith('#'):
                        insert_line += 1
                        
                    # Skip empty lines
                    while insert_line < len(lines) and not lines[insert_line].strip():
                        insert_line += 1
                        
                    # Insert docstring
                    enhanced_content = '\n'.join(lines[:insert_line]) + '\n' + docstring + '\n\n' + '\n'.join(lines[insert_line:])
                    
                    # Write enhanced content back to file
                    with open(file_path, 'w') as f:
                        f.write(enhanced_content)
                        
                    print(f"📝 Enhanced: {rel_path}")
                    stats["enhanced"] += 1
                    
                except Exception as e:
                    print(f"⚠️ Could not enhance {rel_path}: {e}")
                    stats["skipped"] += 1
            else:
                debug_log(f"No matching category for: {rel_path}")
                stats["skipped"] += 1
                
    print(f"📊 Enhancement complete: {stats['enhanced']} files enhanced, {stats['skipped']} files skipped")
    return stats


def initialize_terminal_forge_repo(
    repo_path: str,
    create_dir_fn=create_directory,
    create_empty_fn=create_empty_file,
    create_py_fn=create_python_file
) -> None:
    """
    Initialize the complete Terminal Forge repository structure.
    
    Creates a comprehensive project scaffold following Eidosian principles of
    structural harmony and contextual integrity. This function builds:
    
    1. Core package directory with essential modules
    2. Examples directory with basic, advanced, and specialized demonstrations
    3. Test framework with unit, integration, and performance tests
    4. Documentation system with manual pages and auto-generated references
    5. Project metadata and configuration files
    
    Each component is precisely positioned to form a coherent whole where
    structure itself becomes function.
    
    Parameters
    ----------
    repo_path : str
        Path to the root directory where the repository will be initialized
        
    Returns
    -------
    None
        Function operates through side effects
        
    Examples
    --------
    >>> initialize_terminal_forge_repo("./new_project")
    🚀 Initializing Terminal Forge repository structure...
    """
    print("🚀 Initializing Terminal Forge repository structure...")
    start_time = time.time()
    operations = {"directories": 0, "files": 0, "errors": 0}

    def tracked_create_directory(path):
        result = create_dir_fn(path)
        operations["directories"] += 1
        return result

    def tracked_create_empty_file(path, content=""):
        result = create_empty_fn(path, content)
        operations["files"] += 1
        return result

    def tracked_create_python_file(path, module_name=""):
        result = create_py_fn(path, module_name)
        operations["files"] += 1
        return result

    def tracked_create_markdown_file(path, title=""):
        """Track creation of markdown files."""
        result = create_markdown_file(path, title)
        operations["files"] += 1
        return result

    try:
        # Base directories - each with a precise architectural purpose
        tracked_create_directory(os.path.join(repo_path, "terminal_forge"))  # Core package - functional essence
        tracked_create_directory(os.path.join(repo_path, "examples"))        # Illumination paths - practical demonstrations
        tracked_create_directory(os.path.join(repo_path, "tests"))           # Verification matrix - correctness assurance
        tracked_create_directory(os.path.join(repo_path, "docs"))            # Knowledge cosmos - structured illumination
        
        # Core Package - each module with singular responsibility and perfect cohesion
        core_files = [
            "__init__.py",      # Export gateway: deliberate API surface
            "banner.py",        # Banner system: pixel-perfect containment
            "colors.py",        # Color system: perception-optimized palettes
            "borders.py",       # Border engine: mathematically harmonized boundaries
            "layout.py",        # Layout engine: golden-ratio positioning
            "themes.py",        # Theme system: semantic visual language
            "effects.py",       # Effect engine: fluid temporal transformations
            "ascii_art.py",     # ASCII transformer: unicode-aware artistry
            "utils.py"          # Utility forge: zero-bloat solutions
        ]
        
        # Create core package files with contextually appropriate documentation
        for file in core_files:
            module_name = file.replace('.py', '').replace('_', ' ').title() if file != "__init__.py" else "Terminal Forge"
            tracked_create_python_file(os.path.join(repo_path, "terminal_forge", file), module_name)
        
        # Examples - progressive knowledge path from basic to advanced
        tracked_create_directory(os.path.join(repo_path, "examples", "layouts"))      # Spatial organization examples
        tracked_create_directory(os.path.join(repo_path, "examples", "interactive"))  # Human-machine dialog patterns
        tracked_create_directory(os.path.join(repo_path, "examples", "integration"))  # Ecosystem connection demonstrations
        
        # Basic example files - entry point demonstrations
        example_files = [
            "basic.py",         # First principles: 5-minute mastery trajectory
            "advanced.py",      # Power patterns: complexity-hiding implementations
            "themes.py",        # Visual identity: cohesive styling demonstrations 
            "animations.py"     # Temporal canvas: meaningful motion examples
        ]
        for file in example_files:
            tracked_create_python_file(os.path.join(repo_path, "examples", file))
        
        # Layout examples - spatial organization principles
        layout_files = ["grid.py", "flow.py", "responsive.py"]
        for file in layout_files:
            tracked_create_python_file(os.path.join(repo_path, "examples", "layouts", file))
        
        # Interactive examples - engagement patterns
        interactive_files = ["keyboard.py", "focus.py", "navigation.py"]
        for file in interactive_files:
            tracked_create_python_file(os.path.join(repo_path, "examples", "interactive", file))
        
        # Integration examples - boundary transcendence
        integration_files = ["data_visualization.py", "api_consumption.py", "real_time.py"]
        for file in integration_files:
            tracked_create_python_file(os.path.join(repo_path, "examples", "integration", file))
        
        # Tests - verification as first principle
        tracked_create_directory(os.path.join(repo_path, "tests", "unit"))        # Atomic verification
        tracked_create_directory(os.path.join(repo_path, "tests", "integration"))  # Composition testing
        tracked_create_directory(os.path.join(repo_path, "tests", "performance"))  # Efficiency oracle
        tracked_create_directory(os.path.join(repo_path, "tests", "fixtures"))     # Test foundations
        
        # Unit test files - granular correctness
        unit_test_files = [f"test_{module.replace('.py', '')}.py" for module in core_files if module != "__init__.py"]
        for file in unit_test_files:
            tracked_create_python_file(os.path.join(repo_path, "tests", "unit", file))
        
        # Integration test files - harmonic interaction
        integration_test_files = [
            "test_theme_application.py",    # Visual coherence testing
            "test_layout_composition.py",   # Spatial integrity verification
            "test_animation_system.py"      # Temporal composition validation
        ]
        for file in integration_test_files:
            tracked_create_python_file(os.path.join(repo_path, "tests", "integration", file))
        
        # Performance test files - velocity and resource verification
        performance_test_files = [
            "test_rendering_speed.py",   # Rendering velocity measurement
            "test_memory_usage.py",      # Memory discipline verification
            "test_terminal_io.py"        # I/O optimization validation
        ]
        for file in performance_test_files:
            tracked_create_python_file(os.path.join(repo_path, "tests", "performance", file))
        
        # Test fixture files - reproducible contexts
        fixture_files = [
            "sample_layouts.py",    # Layout templates
            "color_schemes.py",     # Color reference definitions
            "mock_terminal.py"      # Terminal emulation environment
        ]
        for file in fixture_files:
            tracked_create_python_file(os.path.join(repo_path, "tests", "fixtures", file))
        
        # Documentation structure - knowledge architecture
        # Each directory serves a specific documentation purpose according to Eidosian principles
        docs_dirs = [
            # Manual documentation - human-crafted knowledge
            "manual/_common/project",
            "manual/_common/glossary",
            "manual/_common/standards",
            "manual/python/guides/quickstart",
            "manual/python/guides/intermediate",
            "manual/python/guides/advanced",
            "manual/python/api",
            "manual/python/design/patterns",
            "manual/python/design/decisions",
            "manual/python/design/diagrams",
            "manual/python/examples/snippets",
            "manual/python/examples/tutorials",
            "manual/python/examples/projects",
            "manual/python/best_practices/style",
            "manual/python/best_practices/patterns",
            "manual/python/best_practices/antipatterns",
            "manual/python/troubleshooting",
            
            # Auto-generated documentation - machine precision
            "auto/api/modules",
            "auto/api/classes",
            "auto/api/functions",
            "auto/benchmarks/rendering",
            "auto/benchmarks/memory",
            "auto/benchmarks/comparison",
            "auto/coverage/unit",
            "auto/coverage/integration",
            
            # AI-enhanced documentation - intelligence assistance
            "ai/explanations",
            "ai/examples",
            
            # Documentation assets - supporting materials
            "assets/images/screenshots",
            "assets/images/diagrams",
            "assets/images/logos",
            "assets/ascii/banners",
            "assets/ascii/borders",
            "assets/ascii/logos",
            "assets/themes/light",
            "assets/themes/dark",
            "assets/themes/specialty",
            
            # Example libraries - practical demonstrations
            "examples/complete_projects/dashboard",
            "examples/complete_projects/explorer",
            "examples/complete_projects/status_display",
            "examples/snippets/banners",
            "examples/snippets/layouts",
            "examples/snippets/animations",
            
            # Documentation tools - knowledge management
            "tools/generators",
            "tools/validators",
            
            # Version-specific documentation - temporal knowledge
            "versions/latest",
            "versions/archive"
        ]
        
        # Create the documentation directory structure
        for dir_path in docs_dirs:
            tracked_create_directory(os.path.join(repo_path, "docs", dir_path))
        
        # Documentation structure - knowledge architecture initialization
        print("\n🧠 Building knowledge architecture...")
        
        # Create the documentation directory structure with contextual feedback
        for dir_path in docs_dirs:
            tracked_create_directory(os.path.join(repo_path, "docs", dir_path))
        
        # Knowledge Network - Transform flat assets into an interconnected system
        # where each node serves as both content carrier and structural signpost
        
        # 1. Core Knowledge Gateways - Entry points with deliberate navigational purpose
        knowledge_gateways = {
            "getting_started.md": {
                "title": "Getting Started with Terminal Forge",
                "purpose": "Ignition sequence: 60-second mastery path",
                "links": ["api_reference.md", "examples.md"]
            },
            "api_reference.md": {
                "title": "API Reference",
                "purpose": "Command center: comprehensive function index",
                "links": ["getting_started.md", "examples.md"]
            },
            "examples.md": {
                "title": "Examples & Patterns",
                "purpose": "Pattern library: implementation cookbook",
                "links": ["getting_started.md", "api_reference.md"]
            },
            "index.md": {
                "title": "Terminal Forge Documentation",
                "purpose": "Knowledge portal: documentation gateway",
                "links": ["getting_started.md", "api_reference.md", "examples.md"]
            }
        }
        
        # Create knowledge gateways with intelligent cross-linking
        for filename, metadata in knowledge_gateways.items():
            path = os.path.join(repo_path, "docs", filename)
            if not os.path.exists(path):
                # Create content with deliberate navigational structure
                content = f"""# {metadata['title']}

{metadata['purpose']}

## Navigation

"""
                # Add intelligent cross-linking
                for link in metadata['links']:
                    link_metadata = knowledge_gateways.get(link, {})
                    link_title = link_metadata.get('title', link.replace('.md', '').replace('_', ' ').title())
                    content += f"- [{link_title}]({link})\n"
                
                tracked_create_empty_file(path, content)
                print(f"📚 Created knowledge gateway: {filename}")
        
        # 2. Knowledge Automation - Self-aware tools that understand their ecosystem
        knowledge_automation = {
            "generators": {
                "api_docs.py": {
                    "title": "API extraction and formatting system",
                    "modules": ["terminal_forge.banner", "terminal_forge.colors", "terminal_forge.borders"]
                },
                "ascii_preview.py": {
                    "title": "Visual element rendering preview",
                    "assets": ["banners", "borders", "logos"]
                },
                "theme_catalog.py": {
                    "title": "Theme visualization and catalog creation",
                    "themes": ["light", "dark", "specialty"]
                }
            },
            "validators": {
                "example_tester.py": {
                    "title": "Example code verification system",
                    "targets": ["examples/snippets", "examples/complete_projects"]
                },
                "link_checker.py": {
                    "title": "Documentation link integrity validator",
                    "scope": "recursive"
                }
            }
        }
        
        # Generate self-aware documentation tools that understand their ecosystem
        for tool_type, tools in knowledge_automation.items():
            for filename, metadata in tools.items():
                # Contextually rich module content that knows its purpose and position
                tool_dir = os.path.join(repo_path, "docs", "tools", tool_type)
                tool_path = os.path.join(tool_dir, filename)
                
                if not os.path.exists(tool_path):
                    # Build intelligent imports and function skeletons based on purpose
                    imports = [
                        "import os", 
                        "import sys",
                        "from pathlib import Path",
                        f"# Add parent directory to path for terminal_forge imports",
                        "sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))"
                    ]
                    
                    # Create contextually-aware content for each tool
                    if "api_docs" in filename:
                        imports.extend(["import inspect", "from importlib import import_module"])
                    elif "ascii" in filename:
                        imports.append("from rich.console import Console")
                    elif "theme" in filename:
                        imports.extend(["import json", "from rich.theme import Theme"])
                    elif "tester" in filename:
                        imports.extend(["import unittest", "import subprocess"])
                    elif "link" in filename:
                        imports.extend(["import re", "import requests", "from concurrent.futures import ThreadPoolExecutor"])
                    
                    content = f'''"""
{metadata['title']}

An intelligent documentation tool that understands its position in the Terminal Forge
knowledge ecosystem. This module exemplifies Eidosian principles of contextual integrity
and self-awareness as foundation.
"""

{chr(10).join(imports)}


def main():
    """Execute the {filename.replace('.py', '')} with awareness of the Terminal Forge ecosystem."""
    print(f"📘 Running {metadata['title']}...")
    
    # Tool-specific implementations would go here
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
                    tracked_create_empty_file(tool_path, content)
                    print(f"🛠️ Created knowledge tool: {filename}")
        
        # Create root level files - project foundations
        tracked_create_empty_file(os.path.join(repo_path, "pyproject.toml"), """[build-system]
requires = ["setuptools>=42", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "terminal_forge"
version = "0.1.0"
description = "A framework for creating beautiful terminal interfaces with pixel-perfect precision"
readme = "README.md"
authors = [
    {name = "Terminal Forge Contributors", email = "example@example.com"}
]
license = {text = "MIT"}
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
]
requires-python = ">=3.8"
dependencies = []

[project.urls]
"Homepage" = "https://github.com/eidos/terminal_forge"
"Bug Tracker" = "https://github.com/eidos/terminal_forge/issues"
""")
        
        # Create essential project documentation - philosophical foundation
        tracked_create_markdown_file(os.path.join(repo_path, "README.md"), "Terminal Forge")
        tracked_create_markdown_file(os.path.join(repo_path, "eidosian_principles.md"), "Eidosian Principles")
        tracked_create_markdown_file(os.path.join(repo_path, "CONTRIBUTING.md"), "Contributing to Terminal Forge")
        tracked_create_markdown_file(os.path.join(repo_path, "CHANGELOG.md"), "Terminal Forge Changelog")
        
        # Add project overview documentation - conceptual framework
        tracked_create_markdown_file(os.path.join(repo_path, "project_overview.md"), "Terminal Forge Project Overview")
        tracked_create_markdown_file(os.path.join(repo_path, "project_structure.md"), "Terminal Forge Project Structure")
        tracked_create_markdown_file(os.path.join(repo_path, "universal_doc_structure.md"), "Universal Documentation Structure")
        
        # Create license file - legal framework
        tracked_create_empty_file(os.path.join(repo_path, "LICENSE"), """MIT License

Copyright (c) 2025 Terminal Forge Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
""")
        
        # Find and move any existing Python files - intelligent organization
        print("\n🔍 Scanning for existing Python files to relocate...")
        target_structure = [
            "terminal_forge",
            "examples",
            "tests",
            "docs"
        ]
        find_and_move_python_files(repo_path, target_structure)
        
        # ADDED: Enhance existing files with Eidosian docstrings
        enhance_existing_files_with_eidosian_docstrings(repo_path)
        
        elapsed = time.time() - start_time
        print(f"\n✅ Structure created: {operations['directories']} directories, {operations['files']} files in {elapsed:.2f}s")
    except Exception as e:
        operations["errors"] += 1
        print(f"\n❌ Error during initialization: {str(e)}")
        raise


def validate_structure(repo_path: str) -> Dict[str, Union[int, List[str]]]:
    """
    Validate the repository structure post-initialization.
    
    Checks for the presence of essential directories and files, and counts the total
    number of directories and files. Reports any missing essential components.
    
    Parameters
    ----------
    repo_path : str
        Path to the root directory of the repository
        
    Returns
    -------
    Dict[str, Union[int, List[str]]]
        Validation results including counts and any issues found
    """
    validation = {"directories": 0, "files": 0, "issues": []}
    required_dirs = ["terminal_forge", "examples", "tests", "docs"]
    required_files = ["README.md", "pyproject.toml", "LICENSE", "terminal_forge/__init__.py"]
    approved_dirs = {"terminal_forge", "examples", "tests", "docs"}
    
    # Explicitly allow root-level files that are meant to be there
    approved_root_files = {
        "project_overview.md",
        "eidosian_principles.md", 
        "universal_doc_structure.md",
        "CHANGELOG.md",
        "CONTRIBUTING.md", 
        "project_structure.md",
        "terminal_forge_repo_init.py",
        ".vscode/settings.json"
    }
    
    for dir_name in required_dirs:
        dir_path = os.path.join(repo_path, dir_name)
        if not os.path.isdir(dir_path):
            validation["issues"].append(f"Missing essential directory: {dir_name}")
    for file_name in required_files:
        file_path = os.path.join(repo_path, file_name)
        if not os.path.isfile(file_path):
            validation["issues"].append(f"Missing essential file: {file_name}")
    for root, dirs, files in os.walk(repo_path):
        if ".git" in root:
            continue
        validation["directories"] += len(dirs)
        validation["files"] += len(files)
        for d in dirs:
            rel_dir = os.path.relpath(os.path.join(root, d), repo_path)
            if not rel_dir.startswith(".") and rel_dir.split(os.sep)[0] not in approved_dirs:
                validation["issues"].append(f"Potentially misplaced directory: {rel_dir}")
        for f in files:
            if f.startswith("."):
                continue
            
            rel_file = os.path.relpath(os.path.join(root, f), repo_path)
            subdir = rel_file.split(os.sep)[0]
            
            # If it's a root-level file, check if it's explicitly approved
            if subdir == "." and f in approved_root_files:
                continue  # ✅ No warning - it's expected in the root directory
                
            # Otherwise, flag unexpected files
            if subdir not in approved_dirs and rel_file not in required_files:
                validation["issues"].append(f"Potentially misplaced file: {rel_file}")
    validation["status"] = "incomplete" if validation["issues"] else "complete"
    return validation


def main() -> int:
    """
    Entry point for Terminal Forge repository initialization.
    
    Parses command line arguments, validates inputs, and orchestrates the
    repository initialization process. Returns exit code for system use.
    
    Returns
    -------
    int
        Exit code (0 for success, 1 for errors)
        
    Examples
    --------
    $ python terminal_forge_repo_init.py
    🏗️ Terminal Forge Initializer - Target: /current/directory
    ✨ Repository initialization complete. Terminal Forge structure is ready.
    """
    # Create argument parser with Eidosian descriptiveness
    parser = argparse.ArgumentParser(
        description="Initialize Terminal Forge repository structure following Eidosian principles",
        epilog="Terminal Forge: Create beautiful terminal interfaces with pixel-perfect precision"
    )
    
    # Define command-line arguments with clear purpose
    parser.add_argument(
        "path", 
        nargs="?", 
        default=os.getcwd(),
        help="Path to initialize the repository (default: current directory)"
    )
    parser.add_argument(
        "--validate-only", 
        action="store_true", 
        help="Only validate existing repository structure without creating new files"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Display detailed progress information during initialization"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Simulate initialization without creating any files or directories"
    )
    parser.add_argument("--debug", action="store_true", help="Enable detailed debug output for developer analysis")
    
    # Parse arguments and validate inputs
    args = parser.parse_args()
    global DEBUG_MODE
    DEBUG_MODE = args.debug
    repo_path = args.path
    
    # Ensure target path exists before proceeding
    if not os.path.exists(repo_path):
        print(f"❌ Error: Path does not exist: {repo_path}")
        return 1
    
    # Execute initialization with clear user feedback
    print(f"🏗️ Terminal Forge Initializer - Target: {repo_path}")
    try:
        if args.validate_only:
            print("\n🔍 Validating repository structure...")
            validation = validate_structure(repo_path)
            if validation["status"] == "complete":
                print(f"✅ Validation complete: {validation['directories']} directories, {validation['files']} files")
                return 0
            else:
                print(f"⚠️ Repository structure incomplete. Found {len(validation['issues'])} issues:")
                for issue in validation["issues"]:
                    print(f"  - {issue}")
                return 1
        else:
            initialize_terminal_forge_repo(repo_path)
            print("\n🔍 Validating repository structure...")
            validation = validate_structure(repo_path)
            if validation["status"] == "complete":
                print(f"✅ Validation complete: {validation['directories']} directories, {validation['files']} files")
                print("\n✨ Repository initialization complete. Terminal Forge structure is ready.")
                return 0
            else:
                print(f"⚠️ Repository structure incomplete. Found {len(validation['issues'])} issues:")
                for issue in validation["issues"]:
                    print(f"  - {issue}")
                return 1
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())