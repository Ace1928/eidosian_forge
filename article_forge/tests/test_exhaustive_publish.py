import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from article_forge import publish

def test_basic_markdown_fallback():
    """Verify the internal fallback parser works correctly."""
    md = "# Heading\n- Item 1\n- Item 2\n\nParagraph text."
    html = publish._basic_markdown_to_html(md)
    assert "<h1>Heading</h1>" in html
    assert "<ul>" in html
    assert "<li>Item 1</li>" in html
    assert "<p>Paragraph text.</p>" in html

def test_markdown_to_html_structural_wrappers():
    """Ensure the final HTML has proper boilerplate and styling."""
    md = "Hello World"
    html = publish.markdown_to_html(md, title="Custom Title")
    assert "<!doctype html>" in html
    assert "<title>Custom Title</title>" in html
    assert "max-width:860px" in html
    assert "Hello World" in html

def test_convert_file_not_found():
    """Ensure FileNotFoundError is raised for missing sources."""
    with pytest.raises(FileNotFoundError):
        publish.convert_markdown_file("non_existent_file_xyz.md")

def test_convert_file_happy_path(tmp_path):
    """Test the full file conversion flow (HTML only)."""
    src = tmp_path / "test.md"
    src.write_text("# Test Article", encoding="utf-8")
    
    result = publish.convert_markdown_file(src)
    
    assert result.html_path.exists()
    assert "<h1>Test Article</h1>" in result.html_path.read_text()
    assert result.pdf_path is None

@patch("article_forge.publish.html_to_pdf")
def test_convert_file_with_pdf_mock(mock_pdf, tmp_path):
    """Test the full file conversion flow with PDF enabled."""
    src = tmp_path / "test.md"
    src.write_text("Content", encoding="utf-8")
    pdf_out = tmp_path / "out.pdf"
    mock_pdf.return_value = pdf_out
    
    result = publish.convert_markdown_file(src, pdf_out=pdf_out)
    
    assert result.pdf_path == pdf_out
    mock_pdf.assert_called_once()

def test_html_to_pdf_missing_dependency():
    """Verify that a helpful RuntimeError is raised if weasyprint is missing."""
    # We force the import error by patching sys.modules
    with patch.dict("sys.modules", {"weasyprint": None}):
        with pytest.raises(RuntimeError) as exc:
            publish.html_to_pdf("<html></html>", Path("out.pdf"))
        assert "requires 'weasyprint'" in str(exc.value)
