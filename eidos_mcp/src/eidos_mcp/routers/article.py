from __future__ import annotations

from typing import Optional

from eidosian_core import eidosian

from ..core import tool
from ..state import FORGE_DIR

try:
    from article_forge import publish
except ImportError:
    import sys

    sys.path.append(str(FORGE_DIR / "article_forge/src"))
    from article_forge import publish


@eidosian()
@tool(name="article_markdown_to_html", description="Convert Markdown text to a styled HTML document.")
def article_markdown_to_html(markdown_text: str, title: str = "Article") -> str:
    """Transform raw Markdown into a structurally elegant HTML document."""
    return publish.markdown_to_html(markdown_text, title=title)


@eidosian()
@tool(name="article_convert_file", description="Convert a Markdown file to HTML and/or PDF.")
def article_convert_file(source_path: str, html_out: Optional[str] = None, pdf_out: Optional[str] = None) -> str:
    """
    Automates the conversion of Markdown drafts into publishable artifacts.
    Returns a summary of the generated files.
    """
    try:
        result = publish.convert_markdown_file(source_path=source_path, html_out=html_out, pdf_out=pdf_out)
        msg = f"Successfully converted {source_path}."
        if result.html_path:
            msg += f"\nHTML: {result.html_path}"
        if result.pdf_path:
            msg += f"\nPDF: {result.pdf_path}"
        return msg
    except Exception as e:
        return f"Error converting file: {str(e)}"
