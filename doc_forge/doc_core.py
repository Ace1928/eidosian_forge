"""
Doc Forge - Self-evolving documentation system.
Provides contextual awareness and automated documentation generation.
"""

import ast
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from code_forge import CodeProcessor

    HAS_CODE_FORGE = True
except ImportError:
    HAS_CODE_FORGE = False


class DocForge:
    """
    Manages self-evolving documentation.
    Can automatically generate API docs by analyzing source code.
    """

    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path.cwd()
        self.processor = CodeProcessor() if HAS_CODE_FORGE else None

    def generate_readme(self, project_info: Dict[str, Any]) -> str:
        """Generate a structured project README."""
        name = project_info.get("name", "Unknown Project")
        description = project_info.get("description", "No description provided.")
        features = project_info.get("features", [])
        usage = project_info.get("usage", "Usage instructions pending.")

        feature_list = "\n".join([f"- {f}" for f in features])

        return f"# 🔮 {name}\n\n{description}\n\n## 🚀 Features\n\n{feature_list}\n\n## 🛠️ Usage\n\n```python\n{usage}\n```\n"

    def extract_and_generate_api_docs(self, source_dir: Path) -> str:
        """
        Recursively scan a directory and generate API reference from docstrings.
        """
        if not self.processor:
            return "Error: CodeForge not available for analysis."

        modules_data = []
        for file_path in source_dir.rglob("*.py"):
            if file_path.name.startswith("__") and file_path.name != "__init__.py":
                continue

            try:
                source = file_path.read_text()
                tree = self.processor.parse(source)

                module_info = {
                    "name": file_path.relative_to(source_dir),
                    "docstring": ast.get_docstring(tree) or "No module documentation.",
                    "functions": [],
                    "classes": [],
                }

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        module_info["functions"].append(
                            {"name": node.name, "docstring": ast.get_docstring(node) or "No docstring provided."}
                        )
                    elif isinstance(node, ast.ClassDef):
                        module_info["classes"].append(
                            {"name": node.name, "docstring": ast.get_docstring(node) or "No docstring provided."}
                        )

                modules_data.append(module_info)
            except Exception as e:
                print(f"Skipping {file_path}: {e}")

        return self.generate_api_reference(modules_data)

    def generate_api_reference(self, modules: List[Dict[str, Any]]) -> str:
        """Render API reference from module metadata."""
        doc = "# 📚 API Reference\n\n"
        for mod in modules:
            doc += f"## 📄 Module: `{mod['name']}`\n\n{mod['docstring']}\n\n"

            if mod["classes"]:
                doc += "### 🏛️ Classes\n\n"
                for cls in mod["classes"]:
                    doc += f"#### `class {cls['name']}`\n\n{cls['docstring']}\n\n"

            if mod["functions"]:
                doc += "### ⚙️ Functions\n\n"
                for func in mod["functions"]:
                    doc += f"#### `def {func['name']}`\n\n{func['docstring']}\n\n"

            doc += "---\n\n"
        return doc
