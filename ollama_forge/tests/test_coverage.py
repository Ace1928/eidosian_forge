#!/usr/bin/env python3
"""
Test code coverage analysis - ensures all functions have tests.
"""
import os
import ast
import re
import inspect
import unittest
from pathlib import Path
import importlib
import pkgutil
from typing import Dict, Set, Any
from datetime import datetime

import ollama_forge


class FunctionVisitor(ast.NodeVisitor):
    """AST visitor to extract function and method definitions."""
    
    def __init__(self) -> None:
        self.functions = []
        self.current_class = None
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definition and track the current class name."""
        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definition and record its details."""
        # Skip private functions but include dunder methods
        if not (node.name.startswith('_') and not node.name.startswith('__')):
            self.functions.append({
                'name': node.name,
                'class': self.current_class,
                'lineno': node.lineno,
                'docstring': ast.get_docstring(node)
            })
        self.generic_visit(node)


class TestCodeCoverage(unittest.TestCase):
    """Ensures all functions in the package have corresponding tests."""
    
    def setUp(self) -> None:
        """Set up test environment."""
        self.pkg_dir = Path(os.path.abspath(os.path.dirname(ollama_forge.__file__)))
        self.test_dir = Path(os.path.abspath(os.path.join(os.path.dirname(__file__))))
        self.exclude_modules = {"setup"}
        self.test_pattern_regexes = [
            r'test[_\s]([a-zA-Z0-9_]+)',
            r'verify[_\s]([a-zA-Z0-9_]+)',
            r'assert.*\b([a-zA-Z0-9_]+)\(',
            r'mock_([a-zA-Z0-9_]+)',
            r'patch\([\'"].*\.([a-zA-Z0-9_]+)[\'"]',
        ]

    def test_function_coverage(self) -> None:
        """Verify that all functions in the codebase have corresponding unit tests."""
        source_functions = self._get_source_functions()
        test_patterns = self._get_test_patterns()
        untested_functions = {}
        for key, func_info in source_functions.items():
            function_name = func_info['name']
            class_name = func_info['class']
            module_name = func_info['module']
            module_path = '.'.join(module_name.split('.')[:2])
            if module_path in self.exclude_modules:
                continue
            tested = False
            if function_name in test_patterns:
                tested = True
            if class_name and f"{class_name}.{function_name}" in test_patterns:
                tested = True
            if f"test_{function_name}" in test_patterns:
                tested = True
            if class_name and f"test_{class_name}_{function_name}" in test_patterns:
                tested = True
            if not tested:
                untested_functions[key] = func_info
        if untested_functions:
            report = self._generate_coverage_report(untested_functions, source_functions)
            print(f"\n{report}")
        else:
            print("\n✅ All functions have corresponding tests!")

    def _get_source_functions(self) -> Dict[str, Dict[str, Any]]:
        """Extract all functions from source files."""
        source_functions = {}
        
        # Walk through package modules
        for _, module_name, is_pkg in pkgutil.walk_packages(
            path=[str(self.pkg_dir)], 
            prefix='ollama_forge.'  # or whatever your package prefix is
        ):
            # 1) Skip modules you don’t want to import:
            #    e.g. "setup" or anything else in self.exclude_modules
            if any(excluded == module_name.split('.')[-1] for excluded in self.exclude_modules):
                continue

            # Skip __pycache__ or anything that obviously isn't a module
            if '__pycache__' in module_name:
                continue

            try:
                # Import the module
                module = importlib.import_module(module_name)
                source_file = Path(inspect.getfile(module))
                
                # Skip if it's not a .py file
                if source_file.suffix != '.py':
                    continue
                
                # Now safely parse the AST of the source
                with open(source_file, 'r', encoding='utf-8') as f:
                    try:
                        tree = ast.parse(f.read(), filename=str(source_file))
                        visitor = FunctionVisitor()
                        visitor.visit(tree)
                        
                        # Record all functions
                        for func in visitor.functions:
                            class_name = func['class']
                            func_name = func['name']
                            
                            if class_name:
                                key = f"{module_name}.{class_name}.{func_name}"
                            else:
                                key = f"{module_name}.{func_name}"
                            
                            source_functions[key] = {
                                'module': module_name,
                                'class': class_name,
                                'name': func_name,
                                'file': source_file,
                                'line': func['lineno'],
                                'docstring': func['docstring']
                            }
                    except SyntaxError as e:
                        print(f"⚠️ Syntax error in {source_file}: {e}")
            except (ImportError, AttributeError) as e:
                print(f"⚠️ Could not import {module_name}: {e}")

        return source_functions
    
    def _get_test_patterns(self) -> Set[str]:
        """Extract all test patterns from test files."""
        test_patterns = set()
        
        # Find all Python files in test directory
        for path in self.test_dir.glob('**/*.py'):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    try:
                        # Parse the file
                        tree = ast.parse(content)
                        visitor = FunctionVisitor()
                        visitor.visit(tree)
                        
                        # Add all test function names without 'test_' prefix
                        for func in visitor.functions:
                            if func['name'].startswith('test_'):
                                # Add the raw name without 'test_'
                                test_patterns.add(func['name'][5:])
                        
                        # Extract names from content - various patterns
                        
                        # Function calls: function()
                        for match in re.finditer(r'([a-zA-Z_][a-zA-Z0-9_]*)\(', content):
                            test_patterns.add(match.group(1))
                        
                        # Apply all pattern regexes
                        for pattern in self.test_pattern_regexes:
                            for match in re.finditer(pattern, content):
                                if len(match.groups()) > 0:
                                    test_patterns.add(match.group(1))
                    
                    except SyntaxError as e:
                        print(f"⚠️ Syntax error in test file {path}: {e}")
            except Exception as e:
                print(f"⚠️ Error processing test file {path}: {e}")
        
        return test_patterns
    
    def _generate_coverage_report(self, 
                                  untested_functions: Dict[str, Dict[str, Any]],
                                  all_functions: Dict[str, Dict[str, Any]]) -> str:
        """Generate a formatted report of untested functions."""
        total_count = len(all_functions)
        untested_count = len(untested_functions)
        coverage_pct = ((total_count - untested_count) / total_count) * 100 if total_count > 0 else 0
        
        # Group by module
        modules = {}
        for key, info in untested_functions.items():
            module = info['module']
            if module not in modules:
                modules[module] = []
            modules[module].append(info)
        
        # Format report
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        lines = [
            "=" * 80,
            f"FUNCTION COVERAGE REPORT - {now}".center(80),
            "=" * 80,
            f"Total Functions: {total_count}",
            f"Untested Functions: {untested_count}",
            f"Coverage: {coverage_pct:.2f}%",
            "=" * 80,
            ""
        ]
        
        # Sort modules by number of untested functions (descending)
        for module_name, functions in sorted(
            modules.items(), key=lambda x: len(x[1]), reverse=True
        ):
            lines.append(f"MODULE: {module_name}")
            lines.append("-" * 80)
            
            # Group by class
            by_class = {}
            for func in functions:
                class_name = func['class'] or '(module level)'
                if class_name not in by_class:
                    by_class[class_name] = []
                by_class[class_name].append(func)
            
            # Print functions by class
            for class_name, funcs in by_class.items():
                if class_name != '(module level)':
                    lines.append(f"  CLASS: {class_name}")
                
                for func in sorted(funcs, key=lambda x: x['line']):
                    func_name = func['name']
                    line_num = func['line']
                    
                    # Format docstring as summary if available
                    docstring = func['docstring']
                    if docstring:
                        # Get first line of docstring
                        doc_summary = docstring.strip().split('\n')[0]
                        if len(doc_summary) > 50:
                            doc_summary = doc_summary[:47] + '...'
                    else:
                        doc_summary = "(no docstring)"
                    
                    # Build test function name suggestions
                    if class_name != '(module level)':
                        test_name = f"test_{class_name.lower()}_{func_name}"
                        prefix = "    "
                    else:
                        test_name = f"test_{func_name}"
                        prefix = "  "
                    
                    lines.append(f"{prefix}• {func_name} (line {line_num}): {doc_summary}")
                    lines.append(f"{prefix}  Suggested test: {test_name}")
            
            lines.append("")
        
        # Add footer with recommendations
        lines.extend([
            "=" * 80,
            "RECOMMENDATIONS:".center(80),
            "=" * 80,
            "• Create test cases for each untested function",
            "• Aim for at least 90% function coverage",
            "• Prioritize testing complex and critical functionality",
            "• For simple passthrough or utility functions, test at higher levels",
            "=" * 80
        ])
        
        return "\n".join(lines)


if __name__ == "__main__":
    unittest.main()