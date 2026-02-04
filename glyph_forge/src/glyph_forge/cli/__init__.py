"""
⚡ Glyph Forge CLI ⚡

Precision-engineered command line interface for Glyph art transformation.
Zero compromise between power and usability.
"""
import typer
import sys
import os
import time
import logging
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any, Union, Callable

# Initialize logging with zero overhead
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure robust imports that work in all contexts
try:
    from .bannerize import app as bannerize_app
    from .imagize import app as imagize_app
    from .link import app as link_app
    from .audio import app as audio_app
    from .batch import app as batch_app
    from ..config.settings import get_config, ConfigManager
except ImportError as e:
    # Handle case where module is run directly with surgical precision
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    
    try:
        from glyph_forge.cli.bannerize import app as bannerize_app
        from glyph_forge.cli.imagize import app as imagize_app
        from glyph_forge.cli.link import app as link_app
        from glyph_forge.cli.audio import app as audio_app
        from glyph_forge.cli.batch import app as batch_app
        from glyph_forge.config.settings import get_config, ConfigManager
    except ImportError as nested_e:
        logger.critical(f"Failed to import critical modules: {e} -> {nested_e}")
        logger.critical("Please ensure Glyph Forge is correctly installed")
        sys.exit(1)

# Rich library imports for surgical UI precision
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.style import Style
from rich import box as rich_box
import shutil

from .prompting import confirm_action

# Create Typer app with pristine configuration
app = typer.Typer(
    help="⚡ Glyph Forge - Hyper-optimized Glyph art transformation toolkit ⚡",
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Add CLI submodules - no Glyphfy, only imagize (the replacement)
app.add_typer(bannerize_app, name="bannerize", help="Generate stylized text banners")
app.add_typer(imagize_app, name="imagize", help="Transform images into Glyph art masterpieces")
app.add_typer(link_app, name="link", help="Encode/decode share links")
app.add_typer(audio_app, name="audio", help="Audio tools (mux/sync)")
app.add_typer(batch_app, name="batch", help="Batch scan and process videos")

# Initialize console with full capability detection
console = Console()
@app.callback()
def callback():
    """
    Glyph Forge - Where pixels become characters and images transcend their digital boundaries.
    
    The Eidosian engine ensures perfect transformation with zero compromise.
    """
    pass
@app.command()
def version():
    """Display the current version of Glyph Forge with environment details."""
    try:
        from .. import __version__
    except ImportError:
        try:
            from glyph_forge import __version__
        except ImportError:
            __version__ = "unknown"
    
    # Create version table with rich formatting
    table = Table(show_header=False, box=None)
    table.add_column("Property", style="cyan bold")
    table.add_column("Value", style="yellow")
    
    table.add_row("Glyph Forge Version", f"{__version__}")
    table.add_row("Python Version", sys.version.split()[0])
    table.add_row("Platform", f"{sys.platform}")
    
    # Add rich separator
    console.print(Panel("", border_style="bright_yellow", width=60))
    console.print(Panel(Text("⚡ Glyph Forge ⚡", justify="center"), border_style="bright_yellow"))
    console.print(table)
    console.print(Panel("", border_style="bright_yellow", width=60))
@app.command()
def interactive():
    """Launch the interactive Glyph Forge experience."""
    try:
        from textual.app import App
        from ..ui.tui import GlyphForgeApp
        
        console.print("[bold cyan]Launching interactive mode...[/bold cyan]")
        GlyphForgeApp().run()
    except ImportError:
        console.print("[bold red]Error:[/bold red] Textual library not available.")
        console.print("Install with: [bold green]pip install textual[/bold green]")
        console.print("\nFallback to command line mode. Use [bold cyan]--help[/bold cyan] for available commands.")
@app.command()
def stream(
    source: str = typer.Argument(None, help="Video file, YouTube URL, Netflix URL, 'screen', or webcam index"),
    resolution: str = typer.Option("720p", "--resolution", "-r", help="Resolution (1080p/720p/480p)"),
    fps: float = typer.Option(30, "--fps", "-f", help="Target FPS"),
    buffer_seconds: float = typer.Option(30.0, "--buffer-seconds", help="Target buffer duration in seconds"),
    prebuffer_seconds: float = typer.Option(5.0, "--prebuffer-seconds", help="Minimum prebuffer duration in seconds"),
    webcam: int = typer.Option(None, "--webcam", "-w", help="Use webcam with device index"),
    mode: str = typer.Option("gradient", "--mode", "-m", help="Render mode (gradient/braille)"),
    color: str = typer.Option("ansi256", "--color", "-c", help="Color mode (truecolor/ansi256/none)"),
    audio: bool = typer.Option(True, "--audio/--no-audio", help="Enable audio playback"),
    mux_audio: bool = typer.Option(True, "--mux-audio/--no-mux-audio", help="Mux audio into recorded output"),
    stats: bool = typer.Option(True, "--stats/--no-stats", help="Show performance statistics"),
    record: str = typer.Option("auto", "--record", "-o", help="Record glyph output to video (auto/path/none)"),
    output_dir: Optional[str] = typer.Option(None, "--output-dir", help="Directory for output files"),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing outputs"),
    metadata: bool = typer.Option(True, "--metadata/--no-metadata", help="Write metadata sidecar"),
    screen: bool = typer.Option(False, "--screen", "-s", help="Capture screen (for Netflix etc)"),
    duration: float = typer.Option(None, "--duration", "-d", help="Max duration in seconds"),
    render_play: bool = typer.Option(False, "--render-play", help="Render full video with audio muxing, then play result"),
    share: str = typer.Option(None, "--share", help="Export shareable output (mp4/png/gif/apng/webm/html/svg/txt/link)"),
    preset: str = typer.Option(None, "--preset", help="Preset style (cinematic/noir/neon/retro/ultra/filmgrain/vaporwave)"),
    backend: str = typer.Option("auto", "--backend", help="Frame backend (auto/cv2/ffmpeg)"),
    playlist_limit: Optional[int] = typer.Option(None, "--playlist-limit", help="Max playlist items to process"),
    playlist_start: Optional[int] = typer.Option(None, "--playlist-start", help="Playlist start index (1-based)"),
    playlist_end: Optional[int] = typer.Option(None, "--playlist-end", help="Playlist end index (1-based)"),
    yt_cookies: Optional[str] = typer.Option(None, "--yt-cookies", help="Path to cookies.txt for yt-dlp"),
    yt_cookies_from_browser: Optional[str] = typer.Option(
        None,
        "--yt-cookies-from-browser",
        help="Browser cookies spec for yt-dlp (e.g. chrome or firefox:default)",
    ),
    yt_user_agent: Optional[str] = typer.Option(None, "--yt-user-agent", help="Custom User-Agent for yt-dlp"),
    yt_proxy: Optional[str] = typer.Option(None, "--yt-proxy", help="Proxy URL for yt-dlp"),
    yt_skip_authcheck: bool = typer.Option(
        False,
        "--yt-skip-authcheck/--no-yt-skip-authcheck",
        help="Skip YouTube playlist auth check (yt-dlp)",
    ),
    yt_player_client: Optional[str] = typer.Option(
        None,
        "--yt-player-client",
        help="YouTube player client for yt-dlp (e.g. android,web)",
    ),
    yes: Optional[bool] = typer.Option(None, "--yes/--no", "-y/-n", help="Assume yes/no for prompts"),
):
    """Stream video/webcam/YouTube/Netflix/screen to high-fidelity glyph art.
    
    Powered by the Eidosian Unified Stream Engine.
    
    Examples:
        glyph-forge stream video.mp4
        glyph-forge stream https://youtube.com/watch?v=... 
        glyph-forge stream --webcam 0
        glyph-forge stream --screen
    """
    # Handle screen capture mode
    if screen or (source and 'netflix.com' in source.lower()):
        if not confirm_action(
            "Screen capture mode may record sensitive on-screen content. Continue?",
            assume_yes=yes,
            default=False,
        ):
            return
        _stream_screen_capture(
            source,
            duration,
            mode,
            color,
            record,
            stats,
            output_dir=output_dir,
            overwrite=overwrite,
            metadata=metadata,
        )
        return
    
    # Determine source
    actual_source = source
    if webcam is not None:
        actual_source = webcam
    elif source == "webcam":
        actual_source = 0
    elif not source:
        console.print("[bold red]Error:[/bold red] Provide a source or use --webcam/--screen")
        raise typer.Exit(1)

    # Confirm network or playlist sources
    if isinstance(actual_source, str) and actual_source.startswith(("http://", "https://")):
        if not confirm_action(
            f"Stream from URL '{actual_source[:60]}...'?",
            assume_yes=yes,
            default=True,
        ):
            return
        if "list=" in actual_source:
            playlist_note = "Playlist detected. This may download multiple videos."
            if playlist_limit:
                playlist_note += f" Limit: {playlist_limit}."
            if playlist_start or playlist_end:
                playlist_note += f" Range: {playlist_start or 1}-{playlist_end or ''}."
            if not confirm_action(
                f"{playlist_note} Continue?",
                assume_yes=yes,
                default=False,
            ):
                return

    try:
        from ..streaming.engine import UnifiedStreamEngine, UnifiedStreamConfig
        
        render_dithering = True
        render_gamma = 1.15
        render_contrast = 0.98
        render_brightness = 0.02
        render_auto_contrast = True

        # Apply presets
        if preset:
            preset = preset.lower()
            if preset == "cinematic":
                resolution = "720p"
                fps = 30
                mode = "braille"
                color = "ansi256"
                audio = True
            elif preset == "noir":
                resolution = "720p"
                fps = 24
                mode = "gradient"
                color = "none"
                audio = False
                render_dithering = False
                render_gamma = 1.05
                render_contrast = 1.1
                render_brightness = -0.02
            elif preset == "neon":
                resolution = "720p"
                fps = 30
                mode = "braille"
                color = "truecolor"
                audio = True
                render_gamma = 1.2
                render_contrast = 1.0
            elif preset == "retro":
                resolution = "480p"
                fps = 24
                mode = "gradient"
                color = "ansi256"
                audio = True
                render_gamma = 1.05
            elif preset == "ultra":
                resolution = "1080p"
                fps = 60
                mode = "braille"
                color = "truecolor"
                audio = True
                render_dithering = True
            elif preset == "filmgrain":
                resolution = "720p"
                fps = 24
                mode = "gradient"
                color = "ansi256"
                audio = True
                render_dithering = True
                render_gamma = 0.95
                render_contrast = 1.12
                render_brightness = -0.01
            elif preset == "vaporwave":
                resolution = "720p"
                fps = 30
                mode = "braille"
                color = "truecolor"
                audio = True
                render_gamma = 1.25
                render_contrast = 0.95

        # Configure Recording
        rec_enabled = False
        rec_path = None
        if record not in ("none", None) or render_play or share:
            rec_enabled = True
            if record != "auto":
                rec_path = record
            elif render_play or share:
                rec_path = "glyph_forge_render.mp4"

        if yt_cookies and yt_cookies_from_browser:
            console.print("[bold red]Use either --yt-cookies or --yt-cookies-from-browser, not both.[/bold red]")
            raise typer.Exit(1)

        if not yt_cookies:
            yt_cookies = os.environ.get("GLYPH_FORGE_YT_COOKIES")
        if not yt_cookies_from_browser:
            yt_cookies_from_browser = os.environ.get("GLYPH_FORGE_YT_COOKIES_FROM_BROWSER")
        if not yt_user_agent:
            yt_user_agent = os.environ.get("GLYPH_FORGE_YT_USER_AGENT")
        if not yt_proxy:
            yt_proxy = os.environ.get("GLYPH_FORGE_YT_PROXY")
        if not yt_skip_authcheck:
            yt_skip_authcheck = os.environ.get("GLYPH_FORGE_YT_SKIP_AUTHCHECK") == "1"
        if not yt_player_client:
            yt_player_client = os.environ.get("GLYPH_FORGE_YT_PLAYER_CLIENT")

        # Configure Engine
        config = UnifiedStreamConfig(
            source=actual_source,
            resolution=resolution,
            target_fps=int(fps),
            render_mode=mode,
            color_mode=color,
            audio_enabled=audio,
            mux_audio=mux_audio,
            record_enabled=rec_enabled,
            record_path=rec_path,
            output_dir=output_dir,
            overwrite_output=overwrite,
            write_metadata=metadata,
            show_metrics=stats,
            render_then_play=render_play or bool(share),
            play_after_render=render_play,
            max_duration_seconds=duration,
            render_dithering=render_dithering,
            render_gamma=render_gamma,
            render_contrast=render_contrast,
            render_brightness=render_brightness,
            render_auto_contrast=render_auto_contrast,
            buffer_seconds=buffer_seconds,
            prebuffer_seconds=prebuffer_seconds,
            frame_backend=backend,
            playlist_max_items=playlist_limit,
            playlist_start=playlist_start,
            playlist_end=playlist_end,
            yt_cookies=yt_cookies,
            yt_cookies_from_browser=yt_cookies_from_browser,
            yt_user_agent=yt_user_agent,
            yt_proxy=yt_proxy,
            yt_skip_authcheck=yt_skip_authcheck,
            yt_player_client=yt_player_client,
        )
        
        console.print(f"[bold cyan]⚡ GLYPH FORGE STREAMING[/bold cyan]")
        console.print(f"[dim]Source: {actual_source} | Mode: {mode} | Color: {color}[/dim]")
        
        engine = UnifiedStreamEngine(config)
        output_path = engine.run()
        if share:
            from .share_utils import (
                build_share_path,
                export_video_share,
                write_share,
                encode_share_link,
            )
            fmt = share.lower()
            allowed = {"mp4", "png", "gif", "apng", "webm", "html", "svg", "txt", "link"}
            if fmt not in allowed:
                console.print("[bold red]Unsupported share format.[/bold red]")
            elif fmt in {"html", "svg", "txt"}:
                if engine.last_record_text:
                    share_path = build_share_path(
                        str(actual_source),
                        fmt,
                        None,
                        getattr(engine.extraction_info, "title", None),
                    )
                    write_share(engine.last_record_text, fmt, share_path)
                    console.print(f"[green]Share export saved to {share_path}[/green]")
                else:
                    console.print("[bold red]No glyph frame captured for export.[/bold red]")
            elif fmt == "link":
                from ..streaming.naming import build_metadata
                share_path = build_share_path(
                    str(actual_source),
                    "gflink",
                    None,
                    getattr(engine.extraction_info, "title", None),
                )
                if engine.last_record_text:
                    metadata = build_metadata(
                        source=str(actual_source),
                        output_path=share_path,
                        title=getattr(engine.extraction_info, "title", None),
                        info=engine.extraction_info,
                        extra={"content_type": "glyph_text"},
                    )
                    payload = encode_share_link(
                        engine.last_record_text.encode("utf-8"),
                        "txt",
                        "glyph_frame.txt",
                        source=str(actual_source),
                        metadata=metadata,
                    )
                    share_path.write_text(payload, encoding="utf-8")
                    console.print(f"[green]Share link saved to {share_path}[/green]")
                elif output_path:
                    data = Path(output_path).read_bytes()
                    metadata = build_metadata(
                        source=str(actual_source),
                        output_path=share_path,
                        title=getattr(engine.extraction_info, "title", None),
                        info=engine.extraction_info,
                        extra={"content_type": "glyph_video"},
                    )
                    payload = encode_share_link(
                        data,
                        Path(output_path).suffix.lstrip("."),
                        Path(output_path).name,
                        source=str(actual_source),
                        metadata=metadata,
                    )
                    share_path.write_text(payload, encoding="utf-8")
                    console.print(f"[green]Share link saved to {share_path}[/green]")
                else:
                    console.print("[bold red]No output available for link export.[/bold red]")
            elif output_path:
                share_path = build_share_path(
                    str(actual_source),
                    fmt,
                    None,
                    getattr(engine.extraction_info, "title", None),
                )
                if export_video_share(output_path, fmt, share_path):
                    console.print(f"[green]Share export saved to {share_path}[/green]")
                else:
                    console.print("[bold red]Share export failed. Ensure ffmpeg is installed.[/bold red]")
        
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Stream stopped.[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Stream Error:[/bold red] {e}")
        if os.environ.get("GLYPH_FORGE_DEBUG"):
            raise typer.Exit(1)
@app.command()
def demo(
    mode: str = typer.Option("image", "--mode", help="Demo mode (image/banner/stream)"),
    share: str = typer.Option(None, "--share", help="Export shareable output (txt/html/svg)"),
    open_output: bool = typer.Option(False, "--open", help="Open the exported output"),
    copy: bool = typer.Option(False, "--copy", help="Copy banner to clipboard"),
):
    """Run a built-in demo for instant wow."""
    assets_dir = Path(__file__).parent.parent.parent / "assets"
    image_path = assets_dir / "demo.png"
    gif_path = assets_dir / "demo.gif"
    mode = mode.lower()

    if mode == "banner":
        from .. import get_api
        from .share_utils import build_share_path, render_share, try_open_path, copy_to_clipboard
        api = get_api()
        banner = api.generate_banner("GLYPH FORGE", font="slant", style="boxed", color=True)
        if share:
            share_path = build_share_path("glyph_forge_banner", share, None)
            share_path.write_text(render_share(banner, share), encoding="utf-8")
            console.print(f"[green]Share export saved to {share_path}[/green]")
            if open_output and not try_open_path(share_path):
                console.print("[yellow]No opener available on this system.[/yellow]")
        else:
            console.print(banner)
        if copy and not copy_to_clipboard(banner):
            console.print("[yellow]Clipboard copy not available on this system.[/yellow]")
        return

    if mode == "image":
        from .imagize import convert_image
        from .share_utils import build_share_path, render_share, try_open_path
        result = convert_image(
            image_path=str(image_path),
            width=80,
            color_mode="truecolor",
            dithering=True,
            dither_algorithm="atkinson",
            edge_enhance=True,
            sharpen=True,
            optimize_contrast=True,
        )
        if share:
            share_path = build_share_path(str(image_path), share, None)
            share_path.write_text(render_share(result, share), encoding="utf-8")
            console.print(f"[green]Share export saved to {share_path}[/green]")
            if open_output and not try_open_path(share_path):
                console.print("[yellow]No opener available on this system.[/yellow]")
        else:
            console.print(result)
        return

    if mode == "stream":
        if not importlib.util.find_spec("cv2"):
            console.print("[bold red]OpenCV not installed.[/bold red] Install with: pip install opencv-python")
            return
        console.print("[bold cyan]Launching stream demo...[/bold cyan]")
        return stream(str(gif_path))

    console.print("[bold red]Unknown demo mode.[/bold red] Use image, banner, or stream.")
@app.command()
def doctor():
    """Check optional dependencies and system tools."""
    deps = {
        "textual": importlib.util.find_spec("textual") is not None,
        "pillow": importlib.util.find_spec("PIL") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
        "opencv": importlib.util.find_spec("cv2") is not None,
        "pyfiglet": importlib.util.find_spec("pyfiglet") is not None,
        "mss": importlib.util.find_spec("mss") is not None,
        "pygame": importlib.util.find_spec("pygame") is not None,
        "simpleaudio": importlib.util.find_spec("simpleaudio") is not None,
    }
    tools = {
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "ffplay": shutil.which("ffplay") is not None,
        "yt-dlp": shutil.which("yt-dlp") is not None,
        "xdotool": shutil.which("xdotool") is not None,
        "Xvfb": shutil.which("Xvfb") is not None,
    }

    table = Table(title="Glyph Forge Doctor", show_header=True, box=rich_box.ROUNDED)
    table.add_column("Dependency", style="cyan bold")
    table.add_column("Status", style="yellow")
    for name, ok in deps.items():
        table.add_row(name, "✅" if ok else "❌")
    table.add_section()
    for name, ok in tools.items():
        table.add_row(name, "✅" if ok else "❌")
    console.print(table)
@app.command()
def gallery(
    limit: int = typer.Option(30, "--limit", help="Number of assets to render"),
    preset: str = typer.Option("cinematic", "--preset", help="Preset to use for rendering (or 'all')"),
    formats: str = typer.Option("html,png,svg", "--formats", help="Comma-separated output formats (html,png,svg)"),
    upscale: int = typer.Option(4, "--upscale", help="Upscale factor for native-resolution renders (default 4)"),
):
    """Generate a local gallery of glyph renders from the asset library."""
    from pathlib import Path
    import json
    from .imagize import convert_image
    from .share_utils import write_share, build_share_path

    repo_root = Path(__file__).parent.parent.parent
    manifest_path = repo_root / "assets" / "library" / "manifest.json"
    downloads_dir = repo_root / "assets" / "library" / "downloads"
    gallery_dir = repo_root / "gallery"
    if not downloads_dir.exists():
        # Fallback to current working directory (dev runs)
        cwd_root = Path.cwd()
        alt_downloads = cwd_root / "assets" / "library" / "downloads"
        alt_manifest = cwd_root / "assets" / "library" / "manifest.json"
        alt_gallery = cwd_root / "gallery"
        if alt_downloads.exists():
            downloads_dir = alt_downloads
            manifest_path = alt_manifest
            gallery_dir = alt_gallery
    gallery_dir.mkdir(parents=True, exist_ok=True)

    if not downloads_dir.exists():
        console.print("[bold red]Assets not downloaded.[/bold red] Run scripts/download_assets.py")
        raise typer.Exit(1)

    assets = list(downloads_dir.glob("*"))
    images = [a for a in assets if a.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}]
    videos = [a for a in assets if a.suffix.lower() in {".mp4", ".webm", ".mov"}]
    if limit <= 0:
        console.print("[bold red]Limit must be positive.[/bold red]")
        raise typer.Exit(1)
    target_videos = max(0, limit // 2)
    selected = []
    selected.extend(videos[:target_videos])
    remaining = limit - len(selected)
    selected.extend(images[:remaining])
    if len(selected) < limit:
        # Fill from remaining assets if short
        leftovers = [a for a in assets if a not in selected]
        selected.extend(leftovers[: limit - len(selected)])
    assets = selected[:limit]

    html_entries = []
    fmt_set = {f.strip().lower() for f in formats.split(",") if f.strip()}
    presets = [preset] if preset != "all" else ["cinematic", "noir", "neon", "vaporwave", "filmgrain", "ultra"]

    def _extract_frame(video_path: Path) -> Path | None:
        import subprocess
        import shutil
        if not shutil.which("ffmpeg"):
            return None
        out_path = gallery_dir / f"{video_path.stem}_frame.png"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and out_path.exists():
            return out_path
        return None

    for asset in assets:
        suffix = asset.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}:
            source_path = asset
        elif suffix in {".mp4", ".webm", ".mov"}:
            frame = _extract_frame(asset)
            if not frame:
                continue
            source_path = frame
        else:
            continue

        if source_path:
            for preset_name in presets:
                try:
                    from PIL import Image
                    with Image.open(source_path) as img:
                        src_w, src_h = img.size
                    width = min(240, max(80, src_w // 6))
                except Exception:
                    width = 120
                color_mode = "truecolor"
                dithering = True
                dither_algorithm = "atkinson"
                edge_enhance = True
                sharpen = True
                charset = "general"

                if preset_name == "noir":
                    color_mode = "none"
                    edge_enhance = True
                    sharpen = False
                    charset = "blocks"
                elif preset_name == "neon":
                    color_mode = "truecolor"
                    dithering = True
                    dither_algorithm = "floyd-steinberg"
                    charset = "detailed"
                elif preset_name == "vaporwave":
                    color_mode = "truecolor"
                    dithering = True
                    dither_algorithm = "atkinson"
                    charset = "detailed"
                elif preset_name == "filmgrain":
                    color_mode = "ansi256"
                    dithering = True
                    dither_algorithm = "floyd-steinberg"
                    charset = "blocks"
                elif preset_name == "ultra":
                    color_mode = "truecolor"
                    dithering = True
                    dither_algorithm = "atkinson"
                    edge_enhance = True
                    sharpen = True

                result = convert_image(
                    image_path=str(source_path),
                    width=None,
                    height=None,
                    upscale=upscale,
                    color_mode=color_mode,
                    dithering=dithering,
                    dither_algorithm=dither_algorithm,
                    edge_enhance=edge_enhance,
                    sharpen=sharpen,
                    charset=charset,
                )
                if "html" in fmt_set:
                    out_path = gallery_dir / f"{asset.stem}_{preset_name}.html"
                    write_share(result, "html", out_path)
                    html_entries.append((f"{asset.name} ({preset_name})", out_path.name))
                if "png" in fmt_set:
                    png_path = gallery_dir / f"{asset.stem}_{preset_name}.png"
                    write_share(result, "png", png_path)
                if "svg" in fmt_set:
                    svg_path = gallery_dir / f"{asset.stem}_{preset_name}.svg"
                    write_share(result, "svg", svg_path)

    if "html" in fmt_set:
        index = ["<html><body><h1>Glyph Forge Gallery</h1><ul>"]
        for name, path in html_entries:
            index.append(f"<li><a href='{path}'>{name}</a></li>")
        index.append("</ul></body></html>")
        (gallery_dir / "index.html").write_text("\n".join(index), encoding="utf-8")
        console.print(f"[green]Gallery generated: {gallery_dir}/index.html[/green]")
    else:
        console.print(f"[green]Gallery generated: {gallery_dir}[/green]")



def _stream_screen_capture(
    url: Optional[str],
    duration: Optional[float],
    mode: str,
    color: str,
    record: str,
    stats: bool,
    output_dir: Optional[str] = None,
    overwrite: bool = False,
    metadata: bool = True,
):
    """Stream screen capture with optional URL navigation (for Netflix etc.)."""
    try:
        from ..streaming.core.netflix import NetflixCapture, FFmpegCapture, FirefoxController
    except ImportError as e:
        console.print(f"[bold red]Import Error:[/bold red] {e}")
        console.print("Make sure all dependencies are installed.")
        raise typer.Exit(1)
    if not shutil.which("ffmpeg"):
        console.print("[bold red]ffmpeg not available for screen capture.[/bold red]")
        raise typer.Exit(1)
    
    console.print("[bold magenta]═══════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]      🖥️  SCREEN CAPTURE MODE 🖥️         [/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════[/bold magenta]")
    
    # Determine output file
    output_file: Optional[Path]
    if record == "none":
        output_file = None
    elif record == "auto" or record is None:
        from ..streaming.naming import build_output_path
        out_dir = Path(output_dir) if output_dir else None
        output_file = build_output_path(
            source=url or "screen_capture",
            title=None,
            output_dir=out_dir,
            ext="mp4",
            output=None,
            overwrite=overwrite,
        )
    else:
        output_file = Path(record)
    
    # Check if URL provided for Netflix
    if url and 'netflix.com' in url.lower():
        console.print(f"[cyan]Netflix URL detected: {url[:60]}...[/cyan]")
        console.print("[yellow]Will launch Firefox and navigate to URL.[/yellow]")
        
        # Use NetflixCapture for full automation
        firefox = FirefoxController()
        
        console.print("[cyan]Launching Firefox...[/cyan]")
        if firefox.launch():
            console.print("[green]Firefox launched![/green]")
            time.sleep(2)
            
            console.print(f"[cyan]Navigating to Netflix...[/cyan]")
            firefox.navigate(url)
            time.sleep(5)  # Wait for page load
            
            console.print("[yellow]Please log in if needed and start playback.[/yellow]")
            console.print("[bold cyan]Starting capture in 10 seconds...[/bold cyan]")
            console.print("[dim](Press Ctrl+C to cancel)[/dim]")
            
            # Countdown instead of waiting for Enter
            for i in range(10, 0, -1):
                print(f"\r  Starting in {i}s...", end="", flush=True)
                time.sleep(1)
            print()
            
            # Maximize window for best capture
            firefox.fullscreen()
            time.sleep(1)
    
    console.print("[bold green]Starting screen capture...[/bold green]")
    
    # Start FFmpeg capture
    capture_duration = duration or 300  # Default 5 minutes
    ffmpeg = FFmpegCapture(
        output_path=output_file or "/tmp/glyph_screen_capture.mp4",
        width=1920,
        height=1080,
        fps=30
    )
    
    if not ffmpeg.start_recording(capture_duration):
        console.print("[bold red]Failed to start screen capture![/bold red]")
        raise typer.Exit(1)
    
    console.print(f"[green]Recording to: {output_file or '/tmp/glyph_screen_capture.mp4'}[/green]")
    console.print(f"[cyan]Duration: {capture_duration}s (Press Ctrl+C to stop early)[/cyan]")
    
    try:
        # Wait for specified duration or until interrupted
        start_time = time.time()
        while time.time() - start_time < capture_duration:
            elapsed = time.time() - start_time
            remaining = capture_duration - elapsed
            print(f"\r[Recording] {elapsed:.1f}s / {capture_duration}s (Ctrl+C to stop)", end="", flush=True)
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Capture interrupted.[/yellow]")
    finally:
        ffmpeg.stop_recording()
        console.print(f"[bold green]Capture complete![/bold green]")
        if output_file:
            console.print(f"[cyan]Saved to: {output_file}[/cyan]")
            if metadata and output_file.exists():
                from ..streaming.naming import build_metadata, write_metadata
                meta = build_metadata(
                    source=url or "screen_capture",
                    output_path=output_file,
                    title=None,
                    info=None,
                    extra={"capture_duration": capture_duration, "format": "screen_capture"},
                )
                write_metadata(meta, output_file)
            
            # Offer to play the captured file through glyph renderer
            console.print("[yellow]To render as glyph art:[/yellow]")
            console.print(f"[cyan]  glyph-forge stream {output_file}[/cyan]")


@app.command()
def list_commands():
    """Display all available Glyph Forge commands with descriptions."""
    # Create table for commands
    table = Table(title="⚡ Available Commands ⚡", show_header=True, box=rich_box.ROUNDED)
    table.add_column("Command", style="cyan bold")
    table.add_column("Description", style="yellow")
    
    # Add core commands
    table.add_row("version", "Display Glyph Forge version information")
    table.add_row("interactive", "Launch the interactive TUI experience")
    table.add_row("stream", "Stream video/webcam/YouTube to glyph art")
    table.add_row("demo", "Run a built-in demo")
    table.add_row("doctor", "Check optional dependencies and system tools")
    table.add_row("link", "Encode/decode share links")
    table.add_row("audio", "Audio mux/sync tools")
    table.add_row("batch", "Batch scan and process videos")
    table.add_row("list-commands", "Display this command list")
    
    # Add bannerize subcommands
    table.add_section()
    table.add_row("bannerize", "Generate stylized text banners")
    
    # Add imagize subcommands (the replacement for Glyphfy)
    table.add_section()
    table.add_row("imagize", "Transform images into Glyph art masterpieces")
    
    console.print(table)
def main():
    """
    Primary entry point for Glyph Forge CLI
    
    Provides intelligent flow control with perfect error handling
    and optimal user experience in all execution contexts.
    """
    # Environment setup for maximum robustness
    try:
        config = get_config()
    except Exception as e:
        logger.warning(f"Could not load configuration: {e}")
        config = None
    
    # Display the banner when called directly with no arguments
    if len(sys.argv) <= 1:
        display_banner()
        # Fall back to interactive mode for best UX
        return interactive()
    
    # Launch typer app with perfect exception handling
    try:
        return app()
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        if os.environ.get("GLYPH_FORGE_DEBUG"):
            import traceback
            console.print("[bold red]Traceback:[/bold red]")
            console.print(traceback.format_exc())
        return 1
def display_banner():
    """
    Display the Glyph Forge banner with perfect styling
    
    Uses rich formatting for maximum visual impact with
    zero compromise on any terminal environment.
    """
    banner_text = r"""
   ██████╗ ██╗  ██╗   ██╗██████╗ ██╗  ██╗    ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
  ██╔════╝ ██║  ╚██╗ ██╔╝██╔══██╗██║  ██║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
  ██║  ███╗██║   ╚████╔╝ ██████╔╝███████║    █████╗  ██║   ██║██████╔╝██║  ███╗█████╗
  ██║   ██║██║    ╚██╔╝  ██╔═══╝ ██╔══██║    ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
  ╚██████╔╝███████╗██║   ██║     ██║  ██║    ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
   ╚═════╝ ╚══════╝╚═╝   ╚═╝     ╚═╝  ╚═╝    ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝
    """
    # Get terminal width for perfect panel sizing
    width = console.width or 100
    
    # Create styled panel with perfect formatting
    console.print(Panel(
        banner_text, 
        border_style="bright_yellow", 
        title="⚡ Glyph Forge ⚡",
        width=min(width, 100)
    ))
    
    # Print tagline
    console.print("\n[bold yellow]Where pixels become characters and images transcend their digital boundaries.[/bold yellow]")
    console.print("[bold yellow]Powered by Eidosian principles - zero compromise, maximum precision.[/bold yellow]")
    
    # Print usage instructions
    console.print("\nCommands:")
    console.print("  [cyan]glyph-forge imagize[/cyan]   - Transform images to Glyph art")
    console.print("  [cyan]glyph-forge bannerize[/cyan] - Generate text banners")
    console.print("  [cyan]glyph-forge interactive[/cyan] - Launch TUI interface")
    console.print("  [cyan]glyph-forge stream[/cyan]     - Stream video/webcam/YouTube")
    console.print("  [cyan]glyph-forge link[/cyan]       - Encode/decode share links")
    console.print("  [cyan]glyph-forge audio[/cyan]      - Audio mux/sync tools")
    console.print("\nType [cyan]glyph-forge --help[/cyan] for more information\n")
def check_for_external_dependencies() -> Dict[str, bool]:
    """
    Check if optional dependencies are installed with zero IO overhead
    
    Returns:
        Dictionary of dependency availability status
    """
    dependencies = {
        "textual": importlib.util.find_spec("textual") is not None,
        "pillow": importlib.util.find_spec("PIL") is not None,
        "numpy": importlib.util.find_spec("numpy") is not None,
        "opencv": importlib.util.find_spec("cv2") is not None
    }
    
    return dependencies
def get_settings() -> Union[Dict[str, Any], ConfigManager]:
    """
    Compatibility wrapper for settings retrieval with zero friction.
    
    Returns:
        Configuration manager or dictionary
    """
    try:
        return get_config()
    except Exception as e:
        logger.warning(f"Error getting configuration: {e}")
        # Return minimal default settings dictionary
        return {
            "banner": {"default_font": "slant", "default_width": 80},
            "image": {"default_charset": "general", "default_width": 100},
            "io": {"color_output": True}
        }

if __name__ == "__main__":
    sys.exit(main())
