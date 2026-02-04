"""Share link encode/decode commands."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from .prompting import confirm_action
from .share_utils import encode_share_link, load_share_link, write_share, try_open_path
from ..streaming.naming import build_metadata, build_output_path, derive_title, slugify, write_metadata


app = typer.Typer(help="Encode/decode Glyph Forge share links")


@app.command("encode")
def encode(
    source: str = typer.Argument(..., help="File path or raw text to encode"),
    fmt: Optional[str] = typer.Option(None, "--format", "-f", help="Format override (txt/png/gif/mp4/etc)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file for share link"),
    source_ref: Optional[str] = typer.Option(None, "--source", help="Source reference to embed in metadata"),
):
    """Encode a file or text into a glyphforge:// share link."""
    path = Path(source)
    if path.exists():
        data = path.read_bytes()
        fmt = fmt or path.suffix.lstrip(".") or "bin"
        filename = path.name
        source_value = source_ref or str(path)
    else:
        data = source.encode("utf-8")
        fmt = fmt or "txt"
        base = slugify(source[:48])
        filename = f"{base}.{fmt}" if base else f"glyph_link.{fmt}"
        source_value = source_ref

    payload = encode_share_link(data, fmt, filename, source=source_value)
    if output:
        out_path = Path(output)
        out_path.write_text(payload, encoding="utf-8")
        typer.echo(f"Share link saved to {out_path}")
    else:
        typer.echo(payload)


@app.command("decode")
def decode(
    value: str = typer.Argument(..., help="Share link string or .gflink file"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path"),
    fmt: Optional[str] = typer.Option(None, "--format", "-f", help="Override output format"),
    render: bool = typer.Option(False, "--render", help="Render text payloads to PNG"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs"),
    metadata: bool = typer.Option(True, "--metadata/--no-metadata", help="Write metadata sidecar"),
    open_output: bool = typer.Option(False, "--open", help="Open output after decoding"),
    yes: Optional[bool] = typer.Option(None, "--yes/--no", "-y/-n", help="Assume yes/no for prompts"),
):
    """Decode a glyphforge:// share link into a file."""
    payload, data = load_share_link(value)
    payload_fmt = (payload.get("format") or "bin").lower()
    requested_fmt = (fmt or payload_fmt).lower()

    text_formats = {"txt", "html", "svg"}
    if render and payload_fmt in text_formats and fmt is None:
        requested_fmt = "png"

    if output:
        out_path = Path(output)
    else:
        filename = payload.get("filename") or f"glyph_link.{requested_fmt}"
        out_dir = Path.cwd() / "glyph_forge_output"
        out_path = build_output_path(
            source=payload.get("source") or derive_title(filename),
            title=payload.get("title") or Path(filename).stem,
            output_dir=out_dir,
            ext=requested_fmt,
            output=None,
            overwrite=overwrite,
        )

    if out_path.exists() and not overwrite:
        if not confirm_action(
            f"Output exists at {out_path}. Overwrite?",
            assume_yes=yes,
            default=False,
        ):
            return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if payload_fmt in text_formats:
        text = data.decode("utf-8", errors="replace")
        if requested_fmt in text_formats or requested_fmt in {"png", "gif"}:
            write_share(text, requested_fmt, out_path)
        else:
            raise typer.BadParameter(f"Cannot render text payload to {requested_fmt}")
    else:
        if requested_fmt != payload_fmt:
            raise typer.BadParameter(
                f"Binary payload format is {payload_fmt}; cannot decode as {requested_fmt}"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)

    if metadata:
        meta = payload.get("metadata") or {}
        meta.update(
            {
                "source": payload.get("source"),
                "title": payload.get("title") or Path(out_path).stem,
                "format": payload_fmt,
                "decoded_at": datetime.now(timezone.utc).isoformat(),
                "filename": payload.get("filename"),
            }
        )
        write_metadata(build_metadata(meta.get("source") or str(out_path), out_path, title=meta.get("title"), extra=meta), out_path)

    typer.echo(f"Decoded to {out_path}")
    if open_output:
        if not try_open_path(out_path):
            typer.echo("No opener available on this system.")
