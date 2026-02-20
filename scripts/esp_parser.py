from eidosian_core import eidosian

#!/usr/bin/env python3
"""
Eidosian Source Parser (ESP)
Granular symbol extraction for 100% audit coverage.
"""

import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


class ESPParser:
    def __init__(self, audit_root: Path):
        self.audit_root = audit_root
        self.audit_root.mkdir(parents=True, exist_ok=True)
        self.state_file = self.audit_root / "esp_state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, str]:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text())
            except Exception:
                return {}
        return {}

    def _save_state(self):
        self.state_file.write_text(json.dumps(self.state, indent=2))

    def _get_file_hash(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @eidosian()
    def parse_python(self, path: Path) -> List[Dict[str, Any]]:
        symbols = []
        try:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(
                        {"type": "function", "name": node.name, "line": node.lineno, "end_line": node.end_lineno}
                    )
                elif isinstance(node, ast.ClassDef):
                    symbols.append(
                        {"type": "class", "name": node.name, "line": node.lineno, "end_line": node.end_lineno}
                    )
        except Exception as e:
            symbols.append({"type": "error", "name": f"Parse Error: {e}", "line": 0})
        return sorted(symbols, key=lambda x: x["line"])

    @eidosian()
    def generate_checklist(self, source_path: Path, force: bool = False):
        source_path = source_path.resolve()
        file_hash = self._get_file_hash(source_path)
        rel_path = source_path.relative_to(Path.home())
        audit_file = self.audit_root / f"{str(rel_path).replace('/', '_')}.audit.md"

        if not force and self.state.get(str(source_path)) == file_hash and audit_file.exists():
            return  # Idempotent

        symbols = []
        if source_path.suffix == ".py":
            symbols = self.parse_python(source_path)
        else:
            # Fallback for non-python: simple line check
            symbols = [{"type": "file", "name": source_path.name, "line": 1}]

        content = [
            f"# Audit Coverage: {source_path.name}",
            f"**Source**: `{source_path}`",
            f"**Hash**: `{file_hash}`",
            "\n## Symbols Checklist\n",
        ]

        for s in symbols:
            line_ref = f"{source_path}:{s['line']}"
            content.append(f"- [ ] **{s['type'].upper()}**: `{s['name']}` (Line: [{s['line']}]({line_ref}))")

        audit_file.write_text("\n".join(content))
        self.state[str(source_path)] = file_hash
        self._save_state()
        print(f"Generated checklist: {audit_file}")


@eidosian()
def main():
    if len(sys.argv) < 2:
        print("Usage: esp_parser.py <file_or_dir>")
        sys.exit(1)

    target = Path(sys.argv[1])
    audit_dir = Path.home() / ".eidosian/audits/coverage"
    parser = ESPParser(audit_dir)

    if target.is_file():
        parser.generate_checklist(target)
    elif target.is_dir():
        for root, _, files in os.walk(target):
            for f in files:
                if f.endswith((".py", ".sh", ".toml", ".yaml", ".json", ".md")):
                    parser.generate_checklist(Path(root) / f)


if __name__ == "__main__":
    main()
