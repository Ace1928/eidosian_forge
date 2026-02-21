from __future__ import annotations

from pathlib import Path

from article_forge.publish import convert_markdown_file, markdown_to_html


def test_markdown_to_html_renders_heading() -> None:
    html = markdown_to_html("# Title\n\nBody text")
    assert "<h1>Title</h1>" in html
    assert "<p>Body text</p>" in html


def test_convert_markdown_file_writes_html(tmp_path: Path) -> None:
    src = tmp_path / "draft.md"
    src.write_text("# Draft\n\n- alpha\n- beta\n", encoding="utf-8")
    out = tmp_path / "draft.html"

    result = convert_markdown_file(src, html_out=out)

    assert result.source_path == src.resolve()
    assert result.html_path == out.resolve()
    assert result.pdf_path is None
    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "<li>alpha</li>" in html
    assert "<li>beta</li>" in html
