from pathlib import Path

from code_forge.analyzer.generic_analyzer import GenericCodeAnalyzer


def test_generic_analyzer_detects_language() -> None:
    assert GenericCodeAnalyzer.detect_language(Path("main.ts")) == "typescript"
    assert GenericCodeAnalyzer.detect_language(Path("mod.go")) == "go"
    assert GenericCodeAnalyzer.detect_language(Path("notes.unknown")) == "text"


def test_generic_analyzer_extracts_basic_nodes(tmp_path: Path) -> None:
    src = tmp_path / "service.ts"
    src.write_text(
        "class BillingService {\n"
        "  total(items: number[]) {\n"
        "    return items.reduce((acc, n) => acc + n, 0);\n"
        "  }\n"
        "}\n"
        "const calc = (value: number) => value + 1;\n",
        encoding="utf-8",
    )

    analyzer = GenericCodeAnalyzer()
    result = analyzer.analyze_file(src)

    assert result["language"] == "typescript"
    assert result["nodes"]

    names = {node["name"] for node in result["nodes"]}
    unit_types = {node["unit_type"] for node in result["nodes"]}

    assert "BillingService" in names
    assert "class" in unit_types
    assert "calc" in names
    assert "function" in unit_types or "method" in unit_types
