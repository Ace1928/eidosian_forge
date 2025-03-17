from pathlib import Path
import re
import sys
import ast
from typing import Dict, List, Optional, Any, TypedDict, Union, Generator
from datetime import datetime
import json
from .source_discovery import discover_code_structures

# Type definitions
class CodeItem(TypedDict, total=False):
    name: str
    type: str
    file: str
    module_path: str
    parameters: List[str]
    has_docstring: bool
    return_type: Optional[str]
    complexity: int
    class_name: Optional[str]

class CoverageData(TypedDict):
    tested: List[CodeItem]
    untested: List[CodeItem]
    by_module: Dict[str, Dict[str, List[CodeItem]]]

class VizItem(TypedDict, total=False):
    name: str
    children: List['VizItem']
    value: int
    status: str

class TestSuggestion(TypedDict):
    item: CodeItem
    priority: str
    suggested_approach: str
    test_function_name: str

class CodeEntityAnalyzer:
    """
    Advanced code entity analyzer with comprehensive inspection capabilities.
    Discovers and categorizes code structures with Eidosian precision.
    
    Like a master cartographer mapping the neural pathways of your codebase,
    this analyzer reveals the hidden architecture and design patterns that
    give your software its true character. 🧠✨
    """
    
    def discover_all_structures(self) -> List[CodeItem]:
        self.repo_root = repo_root
        self.src_dir = repo_root / "src"
        self.tests_dir = repo_root / "tests"
        self.discovered_items: List[Dict[str, Any]] = []
        self.test_functions: set[str] = set()
        self.module_structure: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    
    def discover_all_structures(self) -> List[Dict[str, Any]]:
        """
        Comprehensively analyze the codebase and discover all code structures.
        Return enriched structure information including parameters, return types, etc.
        
        Like an archaeologist uncovering ancient wisdom, this method extracts
        the deeper truths buried in your source code. 🏺🔍
        """
        # First get the basic structures
        raw_items = discover_code_structures(self.src_dir)
        
        # Enhance items with additional information
        enriched_items: List[Dict[str, Any]] = []
        for item in raw_items:
            # Add module path for better categorization
            file_path = Path(item["file"])
            rel_path = file_path.relative_to(self.repo_root)
            module_path = ".".join(rel_path.with_suffix("").parts[1:])
            
            enriched_item: Dict[str, Any] = {
                **item,
                "module_path": module_path,
                "parameters": [],
                "has_docstring": False,
                "return_type": None,
                "complexity": self._estimate_complexity(file_path, item["name"], item["type"])
            }
            
            # Try to extract more detailed information
            try:
                if file_path.exists():
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if ((item["type"] == "function" and isinstance(node, ast.FunctionDef) or
                             item["type"] == "class" and isinstance(node, ast.ClassDef)) and
                            node.name == item["name"]):
                            
                            # Check for docstring
                            if (node.body and isinstance(node.body[0], ast.Expr) and
                                isinstance(node.body[0].value, ast.Constant) and 
                                isinstance(node.body[0].value.value, str)):
                                enriched_item["has_docstring"] = True
                            
                            # For functions, get parameters and return type hint
                            if isinstance(node, ast.FunctionDef):
                                enriched_item["parameters"] = [arg.arg for arg in node.args.args 
                                                             if arg.arg != 'self' and arg.arg != 'cls']
                                
                                # Try to get return type annotation
                                if node.returns:
                                    enriched_item["return_type"] = ast.unparse(node.returns)
            except Exception:
                pass  # Skip if any issues with detailed analysis
                
            enriched_items.append(enriched_item)
            
        self.discovered_items = enriched_items
        return enriched_items
    
    def analyze_tests(self) -> Dict[str, Any]:
        """
        Analyze existing test files to determine current test coverage
        and create a mapping from code entities to tests.
        """
        self.test_functions = set()
        test_files: Dict[str, Dict[str, Any]] = {}
        
        # Find all test files
        for test_file in self.tests_dir.glob("test_*.py"):
    def _analyze_coverage(self) -> CoverageData:
        """Determine which items are tested and which are not."""
        tested: List[CodeItem] = []
        untested: List[CodeItem] = []
        
        # Group by module for hierarchical analysis
        by_module: Dict[str, Dict[str, List[CodeItem]]] = {}
            test_files[test_file.stem] = {
                "path": test_file,
                "functions": matches,
                "content": content
            }
            
            # Add to global test function set
            for match in matches:
                self.test_functions.add(match.lower())
        
        # Cross-reference items with test functions
        coverage_data = self._analyze_coverage()
        
        return {
            "test_files": test_files,
            "coverage": coverage_data,
            "total_items": len(self.discovered_items),
            "tested_items": len(coverage_data["tested"]),
            "untested_items": len(coverage_data["untested"]),
            "coverage_percentage": self._calculate_coverage_percentage(coverage_data)
        }
    
    def _analyze_coverage(self) -> Dict[str, Any]:
    def _analyze_coverage(self) -> CoverageData:
        """Determine which items are tested and which are not."""
        tested: List[CodeItem] = []
        untested: List[CodeItem] = []
        
        # Group by module for hierarchical analysis
        by_module: Dict[str, Dict[str, List[CodeItem]]] = {}
        
        for item in self.discovered_items:
            # Check both direct name match and module.name match
            module_parts = item["module_path"].split(".")
            item_name = item["name"].lower()
            test_patterns = [
                f"test_{module_parts[-1]}_{item_name}" if module_parts else f"test_{item_name}"
            
            # For classes, also check for test_ClassNameMethod patterns
            if item["type"] == "class":
                test_patterns.append(f"test_{item_name}_")  # Class method tests often start with this
            
            is_tested = any(any(pattern in test_fn for test_fn in self.test_functions) 
                           for pattern in test_patterns)
            
            # Add module info to grouping
            module_name = ".".join(module_parts[:-1]) if len(module_parts) > 1 else "root"
            if module_name not in by_module:
                by_module[module_name] = {"tested": [], "untested": []}
                
            if is_tested:
                tested.append(item)
                by_module[module_name]["tested"].append(item)
            else:
                untested.append(item)
                by_module[module_name]["untested"].append(item)
        
        # Sort items alphabetically by name for consistent output
        tested.sort(key=lambda x: x["name"])
        untested.sort(key=lambda x: x["name"])
        
        self.module_structure = by_module
        
        return {"tested": tested, "untested": untested, "by_module": by_module}
    
    def _calculate_coverage_percentage(self, coverage_data: Dict[str, List[Dict[str, Any]]]) -> float:
        """Calculate test coverage percentage."""
    def _calculate_coverage_percentage(self, coverage_data: CoverageData) -> float:
        """Calculate test coverage percentage."""
        total = len(coverage_data["tested"]) + len(coverage_data["untested"])
        if total == 0:
            return 100.0  # Nothing to test means full coverage
        return (len(coverage_data["tested"]) / total) * 100
        """Estimate item complexity to prioritize tests."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                
            tree = ast.parse(content)
            complexity = 1  # Base complexity
            
            for node in ast.walk(tree):
                if ((item_type == "function" and isinstance(node, ast.FunctionDef) or
                     item_type == "class" and isinstance(node, ast.ClassDef)) and
                    node.name == name):
    def generate_test_suggestions(self) -> Dict[str, Union[List[TestSuggestion], Dict[str, List[TestSuggestion]]]]:
                    # Count branches (if/else statements)
    def generate_test_suggestions(self) -> Dict[str, Union[List[TestSuggestion], Dict[str, List[TestSuggestion]]]]:
        """Generate test suggestions for untested items."""
        # Get coverage data to find untested items
        coverage = self._analyze_coverage()
        untested = coverage["untested"]
        
        # Prioritize untested items by complexity and type
        prioritized = sorted(untested, key=lambda x: (
            -x.get("complexity", 1),  # Higher complexity first
            2 if x["type"] == "class" else 1,  # Classes over functions
            0 if x.get("has_docstring", False) else 1,  # Items with docs are slightly easier to test
            x["name"]  # Alphabetical for consistency
        ))
        
        # Generate suggestions for each untested item
        suggestions: List[TestSuggestion] = []
        for item in prioritized:
            suggestion: TestSuggestion = {
                "item": item,
                "priority": "high" if item.get("complexity", 1) > 5 else "medium" if item.get("complexity", 1) > 2 else "low",
                "suggested_approach": self._suggest_test_approach(item),
                "test_function_name": f"test_{item['name'].lower()}",
            }
            suggestions.append(suggestion)
            
        return {
            "suggestions": suggestions,
            "by_priority": {
                "high": [s for s in suggestions if s["priority"] == "high"],
                "medium": [s for s in suggestions if s["priority"] == "medium"],
                "low": [s for s in suggestions if s["priority"] == "low"]
            }
        }
        prioritized = sorted(untested, key=lambda x: (
            -x.get("complexity", 1),  # Higher complexity first
            2 if x["type"] == "class" else 1,  # Classes over functions
            0 if x.get("has_docstring", False) else 1,  # Items with docs are slightly easier to test
            x["name"]  # Alphabetical for consistency
        ))
        
        # Generate suggestions for each untested item
        suggestions = []
        for item in prioritized:
            suggestion = {
                "item": item,
                "priority": "high" if item.get("complexity", 1) > 5 else "medium" if item.get("complexity", 1) > 2 else "low",
                "suggested_approach": self._suggest_test_approach(item),
                "test_function_name": f"test_{item['name'].lower()}",
            }
            suggestions.append(suggestion)
            
        return {
            "suggestions": suggestions,
            "by_priority": {
        # Group by module for better organization
        by_module: Dict[str, List[CodeItem]] = {}
                "low": [s for s in suggestions if s["priority"] == "low"]
            }
        }
    
    def _suggest_test_approach(self, item: Dict[str, Any]) -> str:
        if item["type"] == "class":
            return f"Create a test class `Test{item['name']}` with setup/teardown methods and test each public method."
        elif "parameters" in item and item["parameters"]:
            param_text = ", ".join(item["parameters"])
            return f"Test with various input combinations for parameters: {param_text}."
        else:
            return f"Create a simple function test that verifies expected behavior."

    def generate_test_stubs(self, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate test stub files for untested items.
        Returns information about generated stubs.
        """
        if not output_dir:
            output_dir = self.tests_dir
            
        if not self.discovered_items:
            self.discover_all_structures()
            
        coverage = self._analyze_coverage()
        untested = coverage["untested"]
        
        # Group by module for better organization
        by_module = {}
        for item in untested:
            module_parts = item["module_path"].split(".")
            module_name = module_parts[0] if module_parts else "core"
            
            if module_name not in by_module:
                by_module[module_name] = []
                
            by_module[module_name].append(item)
        
        # Generate stub files for each module
        generated_files = []
        for module_name, items in by_module.items():
            if not items:
                continue
                
            stub_file = output_dir / f"test_{module_name}_stubs.py"
            
            with open(stub_file, "w", encoding="utf-8") as f:
                # Write imports
                f.write("import unittest\n")
                f.write("import pytest\n")
                
                # Import the module
                f.write(f"import {module_name}\n\n")
                
                # Write test stubs for each item
                for item in items:
                    if item["type"] == "class":
                        f.write(f"class Test{item['name']}(unittest.TestCase):\n")
                        f.write(f"    \"\"\"Tests for {item['name']} class.\"\"\"\n\n")
                        f.write("    def setUp(self):\n")
                        f.write("        # Setup code here\n")
                        f.write("        pass\n\n")
                        f.write(f"    def test_{item['name'].lower()}_initialization(self):\n")
                        f.write(f"        # Test {item['name']} initialization\n")
                        f.write("        self.assertTrue(True)  # Replace with actual test\n\n")
                    else:  # Function
                        f.write(f"def test_{item['name'].lower()}():\n")
                        f.write(f"    \"\"\"Test {item['name']} function.\"\"\"\n")
                        f.write("    # Test implementation here\n")
                        f.write("    assert True  # Replace with actual test\n\n")
                        
            generated_files.append(str(stub_file))
                
        return {
            "generated_files": generated_files,
            "stub_count": len(generated_files),
        # Prepare data structure for visualization
        viz_data: VizItem = {
            "name": "root",
            "children": []
        }
        Generate a visual representation of test coverage as an HTML report.
        
        This creates an interactive sunburst diagram where:
        def visualize_coverage(self, output_path: Path) -> Path:
            """
            Generate a visual representation of test coverage as an HTML report.
            
            This creates an interactive sunburst diagram where:
            - Modules are the first level
            - Types (class/function) are the second level
            - Individual items are the third level
            
            Returns:
                Path to the generated visualization file
            """
                }
            
        # Prepare data structure for visualization
        viz_data = {
            "name": "root",
            "children": []
        }
        
        # Group by modules
        for module_name, data in self.module_structure.items():
            module_node = {
                "name": module_name,
                "children": []
            }
            
            # Group by item type (class or function)
            types = {}
            for status in ["tested", "untested"]:
                for item in data[status]:
                    item_type = item["type"]
                    if item_type not in types:
                        types[item_type] = {"tested": [], "untested": []}
                    types[item_type][status].append(item)
            
            # Add classes
            for item_type, status_data in types.items():
                type_node = {
                    "name": item_type,
                    "children": []
                }
                
                # Add tested items
                for item in status_data["tested"]:
                    type_node["children"].append({
                        "name": item["name"],
                        "value": item.get("complexity", 1),
                        "status": "tested"
                    })
                
                # Add untested items
                for item in status_data["untested"]:
                    type_node["children"].append({
                        "name": item["name"],
                        "value": item.get("complexity", 1),
                        "status": "untested"
                    })
                
                module_node["children"].append(type_node)
            
            viz_data["children"].append(module_node)
        
        # Generate HTML with D3.js visualization
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eidosian Test Coverage Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f9f9f9; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; text-align: center; }
        .stats { display: flex; justify-content: space-around; margin-bottom: 20px; }
        .stat-box { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; width: 200px; }
        .stat-value { font-size: 24px; font-weight: bold; margin: 10px 0; }
        .tested { color: #4CAF50; }
        .untested { color: #F44336; }
        .visualization { display: flex; justify-content: center; }
        .tooltip { position: absolute; background: rgba(0,0,0,0.7); color: white; padding: 5px 10px; border-radius: 3px; font-size: 12px; }
        path { stroke: #fff; }
        path:hover { opacity: 0.8; }
        .tested-item { fill: #4CAF50; }
        .untested-item { fill: #F44336; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Eidosian Test Coverage Visualization</h1>
        <div class="stats">
            <div class="stat-box">
                <div>Total Items</div>
                <div class="stat-value">{total_items}</div>
            </div>
            <div class="stat-box">
                <div>Tested Items</div>
                <div class="stat-value tested">{tested_items}</div>
            </div>
            <div class="stat-box">
                <div>Untested Items</div>
                <div class="stat-value untested">{untested_items}</div>
            </div>
            <div class="stat-box">
                <div>Coverage</div>
                <div class="stat-value">{coverage_percent:.1f}%</div>
            </div>
        </div>
        <div class="visualization">
            <div id="sunburst"></div>
        </div>
    </div>
    
    <script>
        // Visualization data
        const data = {json_data};
        
        // Create sunburst visualization
        const width = 700;
        const height = 700;
        const radius = width / 2;
        
        const arc = d3.arc()
            .startAngle(d => d.x0)
            .endAngle(d => d.x1)
            .padAngle(d => Math.min((d.x1 - d.x0) / 2, 0.005))
            .padRadius(radius / 2)
            .innerRadius(d => d.y0)
            .outerRadius(d => d.y1 - 1);
            
        const partition = data => d3.partition()
            .size([2 * Math.PI, radius])
            (d3.hierarchy(data)
                .sum(d => d.value || 0)
                .sort((a, b) => b.value - a.value));
                
        const root = partition(data);
        
        const svg = d3.select("#sunburst")
            .append("svg")
            .attr("viewBox", [-width / 2, -height / 2, width, height])
            .attr("width", width)
            .attr("height", height)
            .style("font", "10px sans-serif");
            
        const tooltip = d3.select("body").append("div")
            .attr("class", "tooltip")
            .style("opacity", 0);
            
        svg.append("g")
            .selectAll("path")
            .data(root.descendants().filter(d => d.depth))
            .join("path")
            .attr("fill", d => d.data.status === "tested" ? "#4CAF50" : 
                              (d.data.status === "untested" ? "#F44336" : 
                              (d.depth === 1 ? "#2196F3" : "#FFC107")))
            .attr("class", d => d.data.status ? d.data.status + "-item" : "")
            .attr("d", arc)
            .append("title")
            .text(d => `${d.ancestors().map(d => d.data.name).reverse().join(".")}`);
            
        svg.selectAll("path")
            .on("mouseover", function(event, d) {
                tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                tooltip.html(`${d.ancestors().map(d => d.data.name).reverse().join(".")}`)
                    .style("left", (event.pageX + 5) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", function(d) {
                tooltip.transition()
                    .duration(500)
                    .style("opacity", 0);
            });
    </script>
</body>
</html>
""".format(
            total_items=len(self.discovered_items),
            tested_items=len(self._analyze_coverage()["tested"]),
            untested_items=len(self._analyze_coverage()["untested"]),
            coverage_percent=self._calculate_coverage_percentage(self._analyze_coverage()),
            json_data=json.dumps(viz_data)
        )
        
        # Write the HTML file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        return output_path

    def generate_comprehensive_report(self, output_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Generate a comprehensive Eidosian test analysis suite with multiple perspectives.
        
        This method orchestrates the creation of all available reports:
        1. TODO document - What needs testing
        2. Coverage report - Current test status
        3. Test stubs - Starting points for new tests
        4. Visual report - Interactive visualization
        
        Args:
            output_dir: Output directory (defaults to tests directory)
            
        Returns:
            Dictionary mapping report types to file paths
        """
        if output_dir is None:
            output_dir = self.tests_dir
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure we have all necessary data
        if not self.discovered_items:
            self.discover_all_structures()
            
        if not hasattr(self, 'module_structure'):
            self.analyze_tests()
            
        # Generate all reports
        reports = {}
        
        # 1. Generate TODO document
        todo_path = output_dir / "doc_forge_todo.md"
        with open(todo_path, "w", encoding="utf-8") as f:
            f.write("# Doc Forge Comprehensive Test TODO\n\n")
            f.write("This document outlines all testable code structures and their current test status.\n\n")
            
            # Group by module
            by_module = {}
            for item in self.discovered_items:
                module_path = item.get("module_path", "unknown")
                if module_path not in by_module:
                    by_module[module_path] = []
                by_module[module_path].append(item)
            
            # Write by module
            for module, module_items in sorted(by_module.items()):
                f.write(f"## Module: `{module}`\n\n")
                
                # List classes first, then functions, then methods
                classes = [i for i in module_items if i["type"] == "class"]
                functions = [i for i in module_items if i["type"] == "function"]
                methods = [i for i in module_items if i["type"] == "method"]
                
                if classes:
                    f.write("### Classes\n\n")
                    for item in sorted(classes, key=lambda x: x["name"]):
                        params = ", ".join(item.get("parameters", []))
                        has_doc = "✅" if item.get("has_docstring", False) else "❌"
                        is_tested = "✅" if item in self._analyze_coverage()["tested"] else "❌"
                        f.write(f"- [{item['type'].title()}] **{item['name']}** ({params}) | Docstring: {has_doc} | Tested: {is_tested} | Complexity: {item.get('complexity', 1)}\n")
                
                if functions:
                    f.write("\n### Functions\n\n")
                    for item in sorted(functions, key=lambda x: x["name"]):
                        params = ", ".join(item.get("parameters", []))
                        has_doc = "✅" if item.get("has_docstring", False) else "❌"
                        is_tested = "✅" if item in self._analyze_coverage()["tested"] else "❌"
                        f.write(f"- [{item['type'].title()}] **{item['name']}** ({params}) | Docstring: {has_doc} | Tested: {is_tested} | Complexity: {item.get('complexity', 1)}\n")
                
                if methods:
                    f.write("\n### Methods\n\n")
                    for item in sorted(methods, key=lambda x: (x.get("class_name", ""), x["name"])):
                        params = ", ".join(item.get("parameters", []))
                        has_doc = "✅" if item.get("has_docstring", False) else "❌"
                        is_tested = "✅" if item in self._analyze_coverage()["tested"] else "❌"
                        f.write(f"- [{item['type'].title()}] **{item.get('class_name', '')}**.{item['name']} ({params}) | Docstring: {has_doc} | Tested: {is_tested} | Complexity: {item.get('complexity', 1)}\n")
                        
                f.write("\n")
                
        reports["todo"] = todo_path
        
        # 2. Generate coverage report
        coverage_path = output_dir / "doc_forge_coverage.md"
        suggestions = self.generate_test_suggestions()
        with open(coverage_path, "w", encoding="utf-8") as f:
            f.write("# 🔬 Doc Forge Test Coverage Report\n\n")
            f.write("*An Eidosian analysis of test coverage patterns*\n\n")
            
            # Write summary statistics
            coverage_pct = self._calculate_coverage_percentage(self._analyze_coverage())
            f.write(f"## Coverage Summary\n\n")
            
            # Add a visual bar for coverage percentage
            bar_length = 50
            filled_length = int(coverage_pct / 100 * bar_length)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            f.write(f"- **Overall Coverage**: {coverage_pct:.1f}% `{bar}` \n")
            f.write(f"- **Total Items**: {len(self.discovered_items)}\n")
            f.write(f"- **Tested Items**: {len(self._analyze_coverage()['tested'])} ✅\n")
            f.write(f"- **Untested Items**: {len(self._analyze_coverage()['untested'])} ❌\n\n")
            
            # Write module-level coverage
            f.write("## Coverage by Module\n\n")
            for module_name, data in self._analyze_coverage()["by_module"].items():
                total = len(data["tested"]) + len(data["untested"])
                if total == 0:
                    continue
                    
                module_pct = (len(data["tested"]) / total) * 100
                filled_length = int(module_pct / 100 * bar_length)
                bar = "█" * filled_length + "░" * (bar_length - filled_length)
                
                f.write(f"### {module_name}\n\n")
                f.write(f"- **Coverage**: {module_pct:.1f}% `{bar}`\n")
                f.write(f"- **Tested**: {len(data['tested'])}\n")
                f.write(f"- **Untested**: {len(data['untested'])}\n\n")
            
            # Write tested items
            f.write("## Tested Items\n\n")
            for item in self._analyze_coverage()["tested"]:
                f.write(f"- **{item['type'].title()}**: `{item['name']}` ({item['module_path']})\n")
            
            # Write untested items with priority
            f.write("\n## Untested Items\n\n")
            
            # Group by priority
            for priority, symbol in [("high", "🔴"), ("medium", "🟠"), ("low", "🟡")]:
                priority_items = [s for s in suggestions["suggestions"] if s["priority"] == priority]
                if priority_items:
                    f.write(f"### {symbol} {priority.title()} Priority\n\n")
                    for suggestion in priority_items:
                        item = suggestion["item"]
                        f.write(f"- **{item['type'].title()}**: `{item['name']}` ({item['module_path']})\n")
                        f.write(f"  - **Suggestion**: {suggestion['suggested_approach']}\n")
                        f.write(f"  - **Proposed Test Name**: `{suggestion['test_function_name']}`\n\n")
                        
            # Add insights section
            f.write("## 🧠 Eidosian Insights\n\n")
            
            # Calculate statistics
            total_items = len(self.discovered_items)
            total_complexity = sum(i.get("complexity", 1) for i in self.discovered_items)
            avg_complexity = total_complexity / total_items if total_items > 0 else 0
            tested_items = self._analyze_coverage()["tested"]
            tested_complexity = sum(i.get("complexity", 1) for i in tested_items)
            tested_pct = len(tested_items) / total_items * 100 if total_items > 0 else 0
            complexity_pct = tested_complexity / total_complexity * 100 if total_complexity > 0 else 0
            
            # Add insights based on statistics
            f.write(f"- Average code complexity: **{avg_complexity:.2f}**\n")
            
            if complexity_pct < tested_pct:
                f.write("- 🧩 **Complexity Gap**: While {tested_pct:.1f}% of items are tested, only {complexity_pct:.1f}% of complexity is covered.\n")
                f.write("  - *Recommendation*: Focus on testing complex functions first to maximize impact.\n\n")
            else:
                f.write(f"- ✨ **Effective Testing**: {tested_pct:.1f}% of items are tested, covering {complexity_pct:.1f}% of complexity.\n")
                f.write("  - *Observation*: Your tests are effectively targeting complex code paths.\n\n")
            
            # Add recommendations for next steps
            f.write("### 🚀 Next Steps\n\n")
            if suggestions["by_priority"]["high"]:
                high_priority = suggestions["by_priority"]["high"][0]["item"]
                f.write(f"1. Create a test for `{high_priority['name']}` in the `{high_priority['module_path']}` module.\n")
            
            # Add visualization reference
            f.write("\n### 📊 Visualization\n\n")
            f.write("For an interactive visualization of test coverage, open the generated HTML report.\n")
            
        reports["coverage"] = coverage_path
        
        # 3. Generate test stubs
        stubs_info = self.generate_test_stubs(output_dir)
        reports["stubs"] = [Path(p) for p in stubs_info["generated_files"]]
        
        # 4. Generate visualization
        viz_path = self.visualize_coverage(output_dir / "coverage_visualization.html")
        suite_structure: Dict[str, List[CodeItem]] = {}
        
        return reports

    def generate_test_suite(self, output_dir: Optional[Path] = None) -> Path:
        """
        Generate a complete test suite with intelligent structure and organization.
        
        Unlike generate_test_stubs which creates individual stub files, this creates
        a comprehensive test suite with proper organization, import structure, and
        test hierarchy following Eidosian principles.
        
        Args:
            output_dir: Directory to write the test suite (defaults to tests directory)
            
        Returns:
            Path to the generated test suite directory
        """
        if not output_dir:
            output_dir = self.tests_dir / "generated_suite"
            
        # Create the suite directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # If we don't have discovered items yet, run discovery
        if not self.discovered_items:
            self.discover_all_structures()
            
        # Create the suite structure
        suite_structure = {}
        
        # Group by top-level modules
        for item in self.discovered_items:
            module_parts = item["module_path"].split(".")
            top_module = module_parts[0] if module_parts else "core"
            
            if top_module not in suite_structure:
                suite_structure[top_module] = []
                
            suite_structure[top_module].append(item)
        
        # Create __init__.py for the test suite
        with open(output_dir / "__init__.py", "w", encoding="utf-8") as f:
            f.write(f'''"""
Eidosian Test Suite for Doc Forge
Generated on {datetime.now().strftime('%Y-%m-%d at %H:%M')}

This test suite follows Eidosian principles of:
- Structure as Control: Perfect organization of test components
- Flow Like a River: Seamless integration between test modules
- Precision as Style: Each test designed for maximum insight
"""

import unittest
import pytest
import sys
import os
from pathlib import Path

# Add the repository root to the Python path for absolute imports
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
''')
        
        # Create conftest.py with common fixtures
        with open(output_dir / "conftest.py", "w", encoding="utf-8") as f:
            f.write('''"""
Pytest configuration and fixtures for the Eidosian test suite.
"""
import pytest
import os
import sys
from pathlib import Path

# Repository paths for test resources
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "tests"

@pytest.fixture
def repo_root():
    """Return the repository root path."""
    return REPO_ROOT

@pytest.fixture
def sample_docs_dir(tmp_path):
    """Create a temporary docs directory for testing."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    return docs_dir
''')
        
        # Create test modules for each top-level module
        for module_name, items in suite_structure.items():
            module_dir = output_dir / f"test_{module_name}"
            module_dir.mkdir(exist_ok=True)
            
            # Create __init__.py for the module
            with open(module_dir / "__init__.py", "w", encoding="utf-8") as f:
                f.write(f'"""Tests for the {module_name} module."""\n')
            
            # Group by item type
            classes = [i for i in items if i["type"] == "class"]
            functions = [i for i in items if i["type"] == "function"]
            
            # Create test_classes.py
            if classes:
                with open(module_dir / "test_classes.py", "w", encoding="utf-8") as f:
                    f.write(f'''"""
Eidosian tests for {module_name} classes.
Generated on {datetime.now().strftime('%Y-%m-%d')}
"""
import unittest
import pytest
from pathlib import Path

# Import the module being tested
import {module_name}

''')
                    
                    # Add test classes for each class
                    for cls_item in classes:
                        f.write(f'''
class Test{cls_item["name"]}(unittest.TestCase):
    """Test suite for the {cls_item["name"]} class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Initialize test fixtures
        pass
        
    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up after tests
        pass
    
    def test_{cls_item["name"].lower()}_initialization(self):
        """Test {cls_item["name"]} initialization."""
        # Verify the class can be instantiated
        instance = {module_name}.{cls_item["name"]}()  # Adjust constructor parameters as needed
        self.assertIsInstance(instance, {module_name}.{cls_item["name"]})
    
''')
                        
                        # Add known methods if available
                        if "methods" in cls_item:
                            for method in cls_item.get("methods", []):
                                if not method.startswith("__"):  # Skip dunder methods
                                    f.write(f'''    def test_{method.lower()}(self):
        """Test {cls_item["name"]}.{method} method."""
        # Initialize the class
        instance = {module_name}.{cls_item["name"]}()
        
        # Test the method
        # result = instance.{method}()  # Add parameters as needed
        # self.assertTrue(...)  # Add assertions based on expected behavior
        pass
        
''')
            
            # Create test_functions.py
            if functions:
                with open(module_dir / "test_functions.py", "w", encoding="utf-8") as f:
                    f.write(f'''"""
Eidosian tests for {module_name} functions.
Generated on {datetime.now().strftime('%Y-%m-%d')}
"""
import unittest
import pytest
from pathlib import Path

# Import the module being tested
import {module_name}

''')
                    
                    # Add tests for each function
                    for func_item in functions:
                        parameters = func_item.get("parameters", [])
                        param_str = ", ".join("..." for _ in parameters)
                        
                        f.write(f'''
def test_{func_item["name"].lower()}():
    """Test {func_item["name"]} function."""
    # Call the function with appropriate parameters
    # result = {module_name}.{func_item["name"]}({param_str})
    # assert result == ...  # Add assertions based on expected behavior
    pass

''')
                    
                    # Add parameterized tests for functions with parameters
                    for func_item in functions:
                        parameters = func_item.get("parameters", [])
                        if parameters:
                            param_str = ", ".join(p for p in parameters)
                            test_cases_str = ", ".join(f"{p}=..." for p in parameters)
                            
                            f.write(f'''
@pytest.mark.parametrize("{param_str}", [
    ({test_cases_str}),  # Test case 1
    ({test_cases_str}),  # Test case 2
    ({test_cases_str}),  # Test case 3
])
def test_{func_item["name"].lower()}_parametrized({param_str}):
    """Test {func_item["name"]} function with parameterized inputs."""
    # Call the function with the parameterized inputs
    # result = {module_name}.{func_item["name"]}({param_str})
    # assert result == ...  # Add assertions based on expected behavior
    pass

''')
        
        # Create a master test runner
        with open(output_dir / "run_tests.py", "w", encoding="utf-8") as f:
            f.write('''#!/usr/bin/env python3
"""
Eidosian Test Runner - Execute the complete test suite with precision.
"""
import sys
import os
import unittest
import pytest
from pathlib import Path

# Add the repository root to the Python path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Function to run unittest test suite
def run_unittest_suite():
    """Run the complete unittest test suite."""
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent
    suite = loader.discover(start_dir, pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()

# Function to run pytest test suite
def run_pytest_suite():
    """Run the complete pytest test suite."""
    start_dir = Path(__file__).parent
    args = ["-xvs", str(start_dir)]
    return pytest.main(args) == 0

if __name__ == "__main__":
    # Determine which test runner to use
    if len(sys.argv) > 1 and sys.argv[1] == "--pytest":
        success = run_pytest_suite()
    else:
        success = run_unittest_suite()
        
    sys.exit(0 if success else 1)
''')
        
        return output_dir
        analysis: Dict[str, Any] = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "repo": str(self.repo_root),
                "generator": "Eidosian Test Analyzer"
            },
        
        Args:
            output_path: Path to write the JSON file (defaults to tests/analysis.json)
            
        Returns:
            Path to the generated JSON file
        """
        # Ensure we have all necessary data
        if not self.discovered_items:
            self.discover_all_structures()
            
        if not hasattr(self, 'module_structure'):
            self.analyze_tests()
        
        # Create test suggestions if needed
        suggestions = self.generate_test_suggestions()
        
        # Default output path
        if output_path is None:
            output_path = self.tests_dir / "doc_forge_test_analysis.json"
        
        # Build the analysis data structure
        analysis = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "repo": str(self.repo_root),
                "generator": "Eidosian Test Analyzer"
            },
            "summary": {
                "total_items": len(self.discovered_items),
                "tested_items": len(self._analyze_coverage()["tested"]),
                "untested_items": len(self._analyze_coverage()["untested"]),
                "coverage_percentage": self._calculate_coverage_percentage(self._analyze_coverage())
            },
            "modules": {},
            "test_files": {},
            "high_priority_items": [
                {
                    "name": item["name"],
                    "type": item["type"],
                    "module": item["module_path"],
                    "complexity": item.get("complexity", 1),
                    "suggested_test": suggestion["test_function_name"],
                    "suggested_approach": suggestion["suggested_approach"]
                }
                for suggestion, item in [
                    (s, s["item"]) for s in suggestions["by_priority"]["high"]
                ]
            ]
        }
        
        # Add module-level details
        for module_name, data in self._analyze_coverage()["by_module"].items():
            # Skip empty modules
            if not data["tested"] and not data["untested"]:
                continue
                
            total = len(data["tested"]) + len(data["untested"])
            coverage = (len(data["tested"]) / total * 100) if total > 0 else 100
            
            analysis["modules"][module_name] = {
                "coverage_percentage": coverage,
                "tested_count": len(data["tested"]),
                "untested_count": len(data["untested"]),
                "tested_items": [
                    {
                        "name": item["name"],
                        "type": item["type"],
                        "complexity": item.get("complexity", 1),
                        "has_docstring": item.get("has_docstring", False)
                    }
                    for item in data["tested"]
                ],
                "untested_items": [
                    {
                        "name": item["name"],
                        "type": item["type"],
                        "complexity": item.get("complexity", 1),
                        "has_docstring": item.get("has_docstring", False)
                    }
                    for item in data["untested"]
                ]
            }
        
        # Write the JSON file
        with open(output_path, "w", encoding="utf-8") as f:
    print(f"🧪 Test stubs: {len(reports.get('stubs', []))} files")
            json.dump(analysis, f, indent=2)
        
        return output_path
def generate_test_stubs() -> Dict[str, Any]:
    """Generate test stub files for untested items."""
    repo_root = Path(__file__).resolve().parents[1]
    analyzer = CodeEntityAnalyzer(repo_root)
    analyzer.discover_all_structures()
    return analyzer.generate_test_stubs(repo_root / "tests")

if __name__ == "__main__":
    # Process command-line arguments
    if "--all" in sys.argv:
        generate_all_reports()
    elif "--todo" in sys.argv or len(sys.argv) == 1:
        todo_file = generate_todo_document()
        print(f"✅ Generated test TODO document at: {todo_file}")
    elif "--coverage" in sys.argv:
        coverage_file = generate_coverage_report()
        print(f"✅ Coverage report created at: {coverage_file}")
    elif "--stubs" in sys.argv:
        stub_files = generate_test_stubs()
        print(f"✅ Generated {len(stub_files['generated_files'])} test stub files")
    elif "--viz" in sys.argv:
        repo_root = Path(__file__).resolve().parents[1]
        analyzer = CodeEntityAnalyzer(repo_root)
        analyzer.discover_all_structures()
        analyzer.analyze_tests()
        viz_path = analyzer.visualize_coverage()
        print(f"✅ Coverage visualization created at: {viz_path}")
    elif "--help" in sys.argv:
        print("🌀 Eidosian Test Analysis - Available options:")
        print("  --all       Generate all reports (comprehensive analysis)")
        print("  --todo      Generate comprehensive TODO document")
        print("  --coverage  Generate test coverage report")
        print("  --stubs     Generate test stub files")
        print("  --viz       Generate visual coverage report (HTML)")
        print("  --help      Show this help message")
    return reports

if __name__ == "__main__":
    # Process command-line arguments
    if "--all" in sys.argv:
        generate_all_reports()
    elif "--todo" in sys.argv or len(sys.argv) == 1:
        todo_file = generate_todo_document()
        print(f"✅ Generated test TODO document at: {todo_file}")
    elif "--coverage" in sys.argv:
        coverage_file = generate_coverage_report()
        print(f"✅ Coverage report created at: {coverage_file}")
    elif "--stubs" in sys.argv:
        stub_files = generate_test_stubs()
    elif "--viz" in sys.argv:
        repo_root = Path(__file__).resolve().parents[1]
        analyzer = CodeEntityAnalyzer(repo_root)
        analyzer.discover_all_structures()
        analyzer.analyze_tests()
        viz_path = analyzer.visualize_coverage()
        print(f"✅ Coverage visualization created at: {viz_path}")
    elif "--help" in sys.argv:
        print("🌀 Eidosian Test Analysis - Available options:")
        print("  --all       Generate all reports (comprehensive analysis)")
        print("  --todo      Generate comprehensive TODO document")
        print("  --coverage  Generate test coverage report")
        print("  --stubs     Generate test stub files")
        print("  --viz       Generate visual coverage report (HTML)")
        print("  --help      Show this help message")
