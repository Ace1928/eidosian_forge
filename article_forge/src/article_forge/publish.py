"""Publishing helpers for converting Markdown drafts into publishable artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Optional

try:
    import markdown as _markdown_lib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    _markdown_lib = None


@dataclass(frozen=True)
class PublishResult:
    """Result of a publish conversion run."""

    source_path: Path
    html_path: Optional[Path]
    pdf_path: Optional[Path]


def _basic_markdown_to_html(markdown_text: str) -> str:
    """Fallback markdown converter when python-markdown is unavailable."""
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


def markdown_to_html(markdown_text: str, *, title: str = "Article") -> str:
    """Convert markdown text into full HTML document."""
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


def html_to_pdf(html_text: str, output_path: Path) -> Path:
    """Render HTML text to PDF using WeasyPrint if installed."""
    try:
        from weasyprint import HTML  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "PDF export requires 'weasyprint'. Install it in eidosian_venv to enable --pdf-out."
        ) from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=html_text).write_pdf(str(output_path))
    return output_path


def convert_markdown_file(
    source_path: str | Path,
    *,
    html_out: str | Path | None = None,
    pdf_out: str | Path | None = None,
) -> PublishResult:
    """Convert a markdown file into HTML and/or PDF outputs."""
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
