"""
Code Analyzer - Structural Intelligence Engine 📊

Performs deep structural analysis of Python code to discover natural
module boundaries, dependency relationships, and semantic purpose.

Core capabilities:
- AST parsing and traversal with comprehensive node analysis
- Boundary detection using pattern recognition and structural hints
- Dependency graph construction for relationship mapping
- Natural language processing for semantic understanding
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union, Any
import networkx as nx

# ╭──────────────────────────────────────────────────────╮
# │  🔍 Analysis Configuration - Detection Parameters    │
# ╰──────────────────────────────────────────────────────╯

# Patterns that suggest logical module boundaries
MODULE_BOUNDARY_PATTERNS = [
    # Comment-based boundaries
    r"# [\-─═]{2,}.*?[\-─═]{2,}",              # Standard dividers: # ----------
    r"# ╭.*?╮\s*# │.*?│\s*# ╰.*?╯",            # Box dividers
    r"#{3,}\s*([A-Za-z0-9_ ]+)\s*#{3,}",       # Title dividers: ### Section ###
    
    # Code-based boundaries
    r"class [A-Z][a-zA-Z0-9]*(\(.*?\))?:",     # Class definitions  
    r"@\w+\s*(\(.*?\))?\s*\n+def ",            # Decorated function groups
]

# Semantic indicators for module purpose inference
MODULE_PURPOSE_INDICATORS = {
    "util": ["utility", "helper", "tool", "common"],
    "model": ["model", "entity", "data class", "schema", "structure"],
    "service": ["service", "manager", "handler", "processor"],
    "controller": ["controller", "view", "endpoint", "route"],
    "config": ["config", "setting", "parameter", "environment"],
}

# ╭──────────────────────────────────────────────────────╮
# │  🧠 Core Analysis Logic - Structural Intelligence    │
# ╰──────────────────────────────────────────────────────╯

class CodeAnalyzer:
    """Advanced code analyzer that builds comprehensive structural understanding.
    
    Parses Python source into AST and extracts:
    - Structural patterns and natural boundaries
    - Dependency relationships and import requirements
    - Functional groups and their relationships
    - Semantic purpose through comment and docstring analysis
    """
    
    def __init__(self, source_path: Union[str, Path]):
        """Initialize analyzer with path to source file.
        
        Args:
            source_path: Path to Python file for analysis
        """
        self.source_path = Path(source_path)
        if not self.source_path.exists():
            raise FileNotFoundError(f"Source file not found: {self.source_path}")
            
        self.source_code = self.source_path.read_text(encoding='utf-8')
        self.tree = ast.parse(self.source_code)
        # Attach parent links for boundary detection.
        for parent in ast.walk(self.tree):
            for child in ast.iter_child_nodes(parent):
                child.parent = parent
        self.modules = []
        self.dependencies = nx.DiGraph()
        self.symbol_table = {}
    
    def analyze(self) -> Dict[str, Any]:
        """Perform full structural analysis of the source code.
        
        Returns:
            Complete analysis results with structural insights
        """
        # Parse imports and build initial symbol table
        self._analyze_imports()
        
        # Detect logical boundaries in the code
        self._detect_module_boundaries()
        
        # Build dependency graph between identified components
        self._build_dependency_graph()
        
        # Extract documentation and purpose for components
        self._extract_semantic_purpose()
        
        return {
            "modules": self.modules,
            "dependencies": self.dependencies,
            "symbols": self.symbol_table,
            "file_info": {
                "path": str(self.source_path),
                "size": self.source_path.stat().st_size,
                "name": self.source_path.name,
                "stem": self.source_path.stem,
            }
        }
    
    def _analyze_imports(self) -> None:
        """Analyze imports and build symbol table."""
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    self.symbol_table[name.asname or name.name] = {
                        "type": "import",
                        "source": name.name,
                        "alias": name.asname,
                        "lineno": node.lineno
                    }
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                for name in node.names:
                    self.symbol_table[name.asname or name.name] = {
                        "type": "import_from",
                        "source": module,
                        "name": name.name,
                        "alias": name.asname,
                        "lineno": node.lineno
                    }
    
    def _detect_module_boundaries(self) -> None:
        """Detect logical boundaries for module splitting."""
        lines = self.source_code.split('\n')
        
        # Find sections based on boundary patterns
        section_boundaries = []
        for i, line in enumerate(lines):
            for pattern in MODULE_BOUNDARY_PATTERNS:
                if re.match(pattern, line):
                    section_boundaries.append(i)
        
        # Find class and function definitions
        class_funcs = []
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and not node.name.startswith('_'):
                # Top-level definitions only
                if isinstance(node.parent, ast.Module):  
                    class_funcs.append(node.lineno - 1)  # Convert to 0-based index
        
        # Combine and sort all potential boundaries
        all_boundaries = sorted(set(section_boundaries + class_funcs))
        
        # Group closely related boundaries to form coherent modules
        grouped = []
        current_group = []
        
        for boundary in all_boundaries:
            if not current_group or boundary - current_group[-1] < 10:
                current_group.append(boundary)
            else:
                grouped.append(current_group)
                current_group = [boundary]
        
        if current_group:
            grouped.append(current_group)
        
        # Convert grouped boundaries into module definitions
        for i, group in enumerate(grouped):
            start_line = group[0]
            end_line = group[-1]
            
            # Expand to include associated code
            while start_line > 0 and lines[start_line - 1].strip() and not any(re.match(p, lines[start_line - 1]) for p in MODULE_BOUNDARY_PATTERNS):
                start_line -= 1
            
            while end_line < len(lines) - 1 and lines[end_line + 1].strip():
                end_line += 1
                if end_line + 1 < len(lines) and any(re.match(p, lines[end_line + 1]) for p in MODULE_BOUNDARY_PATTERNS):
                    break
            
            # Extract module name and code
            content = '\n'.join(lines[start_line:end_line + 1])
            name = self._extract_module_name(content, f"module_{i}")
            
            self.modules.append({
                "name": name,
                "start_line": start_line,
                "end_line": end_line,
                "content": content
            })
    
    def _extract_module_name(self, content: str, default: str) -> str:
        """Extract appropriate module name from content."""
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

    def _build_dependency_graph(self) -> None:
        """Build a directed graph of dependencies between modules."""
        # Add nodes for all modules
        for module in self.modules:
            self.dependencies.add_node(module["name"], info=module)
        
        # Add dependency edges
        for i, module in enumerate(self.modules):
            module_ast = ast.parse(module["content"])
            
            # Extract all names referenced in this module
            referenced_names = set()
            for node in ast.walk(module_ast):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    referenced_names.add(node.id)
            
            # Check if names are defined in other modules
            for j, other_module in enumerate(self.modules):
                if i == j:
                    continue
                    
                other_ast = ast.parse(other_module["content"])
                defined_names = set()
                
                for node in ast.walk(other_ast):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                        defined_names.add(node.name)
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                defined_names.add(target.id)
                
                # If this module references names defined in the other module, add dependency
                if referenced_names.intersection(defined_names):
                    self.dependencies.add_edge(module["name"], other_module["name"])
    
    def _extract_semantic_purpose(self) -> None:
        """Extract semantic purpose and documentation from the code."""
        for module in self.modules:
            content = module["content"]
            
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
                    inferred_purpose = self._infer_purpose(content, module["name"])
                    module["purpose"] = inferred_purpose
            
            # Extract functions and classes
            module_ast = ast.parse(content)
            module["functions"] = []
            module["classes"] = []
            
            for node in ast.iter_child_nodes(module_ast):
                if isinstance(node, ast.FunctionDef):
                    module["functions"].append({
                        "name": node.name,
                        "lineno": node.lineno,
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node) or ""
                    })
                elif isinstance(node, ast.ClassDef):
                    module["classes"].append({
                        "name": node.name,
                        "lineno": node.lineno,
                        "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                        "docstring": ast.get_docstring(node) or ""
                    })
    
    def _infer_purpose(self, content: str, name: str) -> str:
        """Infer the purpose of a module from its content and name."""
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


# ╭──────────────────────────────────────────────────────╮
# │  🔌 Public Interface - Analysis Entry Points         │
# ╰──────────────────────────────────────────────────────╯

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
    results = analyzer.analyze()
    return results


def print_analysis_report(results: Dict[str, Any]) -> None:
    """Print a human-readable report from analysis results.
    
    Args:
        results: Analysis results from analyze_code()
    """
    print(f"📊 Analysis Report for {results['file_info']['path']}")
    print(f"Found {len(results['modules'])} potential modules:")
    
    for i, module in enumerate(results['modules'], 1):
        print(f"\n{i}. {module['name']}")
        print(f"   Purpose: {module.get('purpose', 'Unknown purpose')}")
        print(f"   Lines: {module['start_line']+1}-{module['end_line']+1}")
        print(f"   Functions: {', '.join(f['name'] for f in module.get('functions', []))}")
        print(f"   Classes: {', '.join(c['name'] for c in module.get('classes', []))}")
        
        # Print dependencies if any
        deps = []
        for _, target in results['dependencies'].out_edges(module['name']):
            deps.append(target)
        
        if deps:
            print(f"   Depends on: {', '.join(deps)}")
