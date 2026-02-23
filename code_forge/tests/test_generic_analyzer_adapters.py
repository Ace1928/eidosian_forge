from pathlib import Path

from code_forge.analyzer.generic_analyzer import GenericCodeAnalyzer
from code_forge.analyzer.parser_adapters import TreeSitterAdapter


class _MockAdapter:
    def __init__(self) -> None:
        self.calls = 0

    def supports_language(self, language: str) -> bool:
        return language == "javascript"

    def analyze_file(self, file_path: Path, source: str, language: str):
        self.calls += 1
        return {
            "language": language,
            "classes": [],
            "functions": [
                {
                    "name": "via_adapter",
                    "docstring": None,
                    "source": "function via_adapter() {}",
                    "args": [],
                    "line_start": 1,
                    "line_end": 1,
                    "col_start": 0,
                    "col_end": 24,
                }
            ],
            "imports": [],
            "docstring": None,
            "module": {
                "docstring": None,
                "source": source,
                "line_start": 1,
                "line_end": 1,
                "col_start": 0,
                "col_end": 0,
            },
            "nodes": [],
            "edges": [],
            "parser_adapter": "mock",
        }


class _NullAdapter:
    def supports_language(self, language: str) -> bool:
        return language == "javascript"

    def analyze_file(self, file_path: Path, source: str, language: str):
        return None


def test_generic_analyzer_prefers_adapter_result(tmp_path: Path) -> None:
    file_path = tmp_path / "index.js"
    file_path.write_text("function fallback() { return 1; }\n", encoding="utf-8")
    adapter = _MockAdapter()
    analyzer = GenericCodeAnalyzer(adapters=[adapter])

    result = analyzer.analyze_file(file_path)

    assert result.get("parser_adapter") == "mock"
    assert adapter.calls == 1
    fn_names = {entry.get("name") for entry in result.get("functions", [])}
    assert "via_adapter" in fn_names


def test_generic_analyzer_falls_back_when_adapter_returns_none(tmp_path: Path) -> None:
    file_path = tmp_path / "index.js"
    file_path.write_text("function fallback() { return 1; }\n", encoding="utf-8")
    analyzer = GenericCodeAnalyzer(adapters=[_NullAdapter()])

    result = analyzer.analyze_file(file_path)

    assert "parser_adapter" not in result
    fn_names = {entry.get("name") for entry in result.get("functions", [])}
    assert "fallback" in fn_names


def test_tree_sitter_adapter_is_safe_when_runtime_missing() -> None:
    adapter = TreeSitterAdapter()
    # Adapter must never crash callers; availability depends on optional deps.
    assert isinstance(adapter.available, bool)
    if not adapter.available:
        assert adapter.analyze_file(Path("x.js"), "function x(){}", "javascript") is None
