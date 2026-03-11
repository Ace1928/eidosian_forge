"""Publishing helpers for converting Markdown drafts into publishable artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Optional

from eidosian_core import eidosian

try:
    import markdown as _markdown_lib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _markdown_lib = None


@dataclass(frozen=True)
class PublishResult:
    """
    Standardized result of an Eidosian publishing run.
    
    Attributes:
        source_path (Path): The original Markdown source.
        html_path (Optional[Path]): Path to the generated HTML artifact.
        pdf_path (Optional[Path]): Path to the generated PDF artifact.
    """

    source_path: Path
    html_path: Optional[Path]
    pdf_path: Optional[Path]


def _basic_markdown_to_html(markdown_text: str) -> str:
    """
    Internal fallback converter for environments lacking the full markdown library.
    Implements a subset of Eidosian structural conventions (headings, lists, paragraphs).
    """
    html_parts: list[str] = []
    in_list = False
    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            continue
        if line.startswith("#"):
            if in_list:
                html_parts.append("</ul>")
                in_list = False
            level = min(len(line) - len(line.lstrip("#")), 6)
            content = escape(line[level:].strip())
            html_parts.append(f"<h{level}>{content}</h{level}>")
            continue
        if line.startswith(("- ", "* ")):
            if not in_list:
                html_parts.append("<ul>")
                in_list = True
            html_parts.append(f"<li>{escape(line[2:].strip())}</li>")
            continue
        if in_list:
            html_parts.append("</ul>")
            in_list = False
        html_parts.append(f"<p>{escape(line)}</p>")
    if in_list:
        html_parts.append("</ul>")
    return "\n".join(html_parts)


@eidosian()
def markdown_to_html(markdown_text: str, *, title: str = "Article") -> str:
    """
    Transforms raw Markdown into a structurally elegant, Eidosian-styled HTML document.
    
    Args:
        markdown_text (str): The source text to render.
        title (str): Document title for metadata and header.
        
    Returns:
        str: A complete, self-contained HTML5 document.
    """
    if _markdown_lib is not None:
        body = _markdown_lib.markdown(
            markdown_text,
            extensions=["extra", "sane_lists", "tables"],
            output_format="html5",
        )
    else:
        body = _basic_markdown_to_html(markdown_text)
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\" />\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />\n"
        f"  <title>{escape(title)}</title>\n"
        "  <style>body{font-family:system-ui,Arial,sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem;line-height:1.6;}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{body}\n"
        "</body>\n"
        "</html>\n"
    )


@eidosian()
def html_to_pdf(html_text: str, output_path: Path) -> Path:
    """
    High-fidelity rendering of HTML into PDF format.
    
    Requires 'weasyprint' to be present in the active environment.
    
    Args:
        html_text (str): The source HTML to render.
        output_path (Path): Destination path for the PDF artifact.
        
    Returns:
        Path: The absolute path to the generated PDF.
    """
    try:
        from weasyprint import HTML  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "PDF export requires 'weasyprint'. Install it in eidosian_venv to enable --pdf-out."
        ) from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_text).write_pdf(str(output_path))
    return output_path


@eidosian()
def convert_markdown_file(
    source_path: str | Path,
    *,
    html_out: str | Path | None = None,
    pdf_out: str | Path | None = None,
) -> PublishResult:
    """
    The primary entry point for the Eidosian Article Pipeline.
    Automates the conversion of Markdown drafts into publishable HTML and PDF artifacts.
    
    Args:
        source_path (str | Path): Path to the Markdown source file.
        html_out (Optional[str | Path]): Explicit output path for HTML.
        pdf_out (Optional[str | Path]): Explicit output path for PDF.
        
    Returns:
        PublishResult: Metadata containing paths to all generated artifacts.
    """
    src = Path(source_path).expanduser().resolve()
    if not src.exists():
        raise FileNotFoundError(f"Markdown source not found: {src}")

    markdown_text = src.read_text(encoding="utf-8")
    html_text = markdown_to_html(markdown_text, title=src.stem.replace("_", " ").title())

    html_path: Optional[Path] = None
    pdf_path: Optional[Path] = None

    if html_out is not None:
        html_path = Path(html_out).expanduser().resolve()
    elif pdf_out is None:
        html_path = src.with_suffix(".html")

    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html_text, encoding="utf-8")

    if pdf_out is not None:
        pdf_path = html_to_pdf(html_text, Path(pdf_out).expanduser().resolve())

    return PublishResult(source_path=src, html_path=html_path, pdf_path=pdf_path)
