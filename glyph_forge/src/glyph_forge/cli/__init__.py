"""
⚡ Glyph Forge CLI ⚡

Precision-engineered command line interface for Glyph art transformation.
Zero compromise between power and usability.
"""
from eidosian_core import eidosian
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
    from ..config.settings import get_config, ConfigManager
except ImportError as e:
    # Handle case where module is run directly with surgical precision
    current_dir = Path(__file__).parent
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))
    
    try:
        from glyph_forge.cli.bannerize import app as bannerize_app
        from glyph_forge.cli.imagize import app as imagize_app
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

# Create Typer app with pristine configuration
app = typer.Typer(
    help="⚡ Glyph Forge - Hyper-optimized Glyph art transformation toolkit ⚡",
    add_completion=True,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Add CLI submodules - no Glyphfy, only imagize (the replacement)
app.add_typer(bannerize_app, name="bannerize", help="Generate stylized text banners")
app.add_typer(imagize_app, name="imagize", help="Transform images into Glyph art masterpieces")

# Initialize console with full capability detection
console = Console()

@eidosian()
@app.callback()
def callback():
    """
    Glyph Forge - Where pixels become characters and images transcend their digital boundaries.
    
    The Eidosian engine ensures perfect transformation with zero compromise.
    """
    pass

@eidosian()
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

@eidosian()
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

@eidosian()
@app.command()
def stream(
    source: str = typer.Argument(None, help="Video file, YouTube URL, Netflix URL, 'screen', or webcam index"),
    resolution: str = typer.Option("auto", "--resolution", "-r", help="Resolution (1080p/720p/480p/auto)"),
    fps: float = typer.Option(None, "--fps", "-f", help="Target FPS (None = match source)"),
    webcam: int = typer.Option(None, "--webcam", "-w", help="Use webcam with device index"),
    mode: str = typer.Option("gradient", "--mode", "-m", help="Render mode (gradient/braille/blocks)"),
    color: str = typer.Option("ansi256", "--color", "-c", help="Color mode (truecolor/ansi256/none)"),
    audio: bool = typer.Option(True, "--audio/--no-audio", help="Enable audio playback"),
    stats: bool = typer.Option(True, "--stats/--no-stats", help="Show performance statistics"),
    record: str = typer.Option("auto", "--record", "-o", help="Record glyph output to video (auto/path/none)"),
    force: bool = typer.Option(False, "--force", help="Force re-render even if cached"),
    legacy: bool = typer.Option(False, "--legacy", help="Use legacy streaming engine"),
    core: bool = typer.Option(False, "--core", help="Use new modular core engine"),
    screen: bool = typer.Option(False, "--screen", "-s", help="Capture screen (for Netflix etc)"),
    duration: float = typer.Option(None, "--duration", "-d", help="Max duration in seconds (limits playback/recording)"),
):
    """Stream video/webcam/YouTube/Netflix/screen to high-fidelity glyph art.
    
    ULTIMATE DEFAULTS (v3):
    - Auto resolution (fits to terminal)
    - Match source FPS (or 30 if unknown)
    - Gradient rendering (best visual quality)
    - ANSI256 colors (good quality + performance)
    - 5-60s prebuffer (auto-calculated for smooth playback)
    - Recording enabled (saves glyph output as video!)
    
    Examples:
        glyph-forge stream video.mp4                    # Ultimate defaults + auto record
        glyph-forge stream https://youtube.com/watch?v=... 
        glyph-forge stream --webcam 0 --record webcam_art.mp4
        glyph-forge stream video.mp4 --mode braille --color truecolor
        glyph-forge stream video.mp4 --record none     # No recording
        glyph-forge stream https://netflix.com/watch/... --screen  # Netflix via screen capture
        glyph-forge stream --screen -d 60              # Capture screen for 60 seconds
    """
    # Handle screen capture mode for Netflix etc.
    if screen or (source and 'netflix.com' in source.lower()):
        _stream_screen_capture(source, duration, mode, color, record, stats)
        return
    
    # Handle webcam option
    if webcam is not None:
        actual_source = webcam
    elif source == "webcam":
        actual_source = 0
    elif source == "screen":
        _stream_screen_capture(None, duration, mode, color, record, stats)
        return
    elif source:
        actual_source = source
    else:
        console.print("[bold red]Error:[/bold red] Provide a source (video/URL) or use --webcam or --screen")
        raise typer.Exit(1)
    
    # Use legacy mode if requested
    if legacy:
        _stream_legacy(actual_source, fps or 30, 1, "standard", "sobel", color != "none")
        return
    
    # Use new modular core engine if requested
    if core:
        try:
            from ..streaming.core import GlyphStreamEngine, StreamConfig, RenderMode, ColorMode
            
            # Map mode string to enum
            mode_map = {
                'gradient': RenderMode.GRADIENT,
                'braille': RenderMode.BRAILLE,
                'ascii': RenderMode.ASCII,
                'blocks': RenderMode.BLOCKS,
            }
            color_map = {
                'truecolor': ColorMode.TRUECOLOR,
                'ansi256': ColorMode.ANSI256,
                'none': ColorMode.NONE,
            }
            
            config = StreamConfig(
                render_mode=mode_map.get(mode, RenderMode.GRADIENT),
                color_mode=color_map.get(color, ColorMode.ANSI256),
                target_fps=fps or 0,
                audio_enabled=audio,
                record_output=record != "none",
                output_path=record if record not in ("auto", "none") else None,
                force_rerender=force,
                show_metrics=stats,
            )
            
            console.print(f"[bold cyan]🎬 GLYPH FORGE CORE ENGINE[/bold cyan]")
            console.print(f"[dim]Mode: {mode} | Color: {color} | Record: {record}[/dim]")
            
            engine = GlyphStreamEngine(config)
            engine.stream(str(actual_source))
            return
            
        except Exception as e:
            console.print(f"[bold yellow]Core engine error:[/bold yellow] {e}")
            console.print("[dim]Falling back to ultimate engine...[/dim]")
    
    # Use ULTIMATE streaming engine (best quality)
    try:
        from ..streaming.ultimate import UltimateConfig, UltimateStreamEngine, generate_output_name
        
        # Determine record path
        if record == "none":
            record_enabled = False
            record_path = None
        elif record == "auto":
            record_enabled = True
            record_path = None  # Auto-generate
        else:
            record_enabled = True
            record_path = record
        
        config = UltimateConfig(
            resolution=resolution,
            target_fps=fps,  # None = match source
            render_mode=mode,
            color_mode=color,
            audio_enabled=audio,
            record_enabled=record_enabled,
            force_rerender=force,
            show_metrics=stats,
        )
        
        console.print(f"[bold cyan]🎬 GLYPH FORGE ULTIMATE[/bold cyan]")
        console.print(f"[dim]Resolution: {resolution} | Mode: {mode} | Color: {color}[/dim]")
        console.print(f"[dim]Recording: {'auto' if record == 'auto' else record or 'disabled'} | Force: {force}[/dim]")
        
        engine = UltimateStreamEngine(config)
        engine.run(actual_source, output_path=record_path, force=force, max_duration=duration)
        
    except ImportError as e:
        # Fallback to premium engine
        console.print(f"[bold yellow]Warning:[/bold yellow] Ultimate engine not available ({e})")
        console.print("[dim]Using premium streaming engine...[/dim]")
        
        try:
            from ..streaming.premium import PremiumConfig, PremiumStreamEngine
            
            config = PremiumConfig(
                resolution=resolution,
                target_fps=int(fps) if fps else 30,
                render_mode=mode,
                color_mode=color,
                audio_enabled=audio,
                record_enabled=record not in (None, "none"),
                record_path=record if record not in ("auto", "none") else None,
                show_metrics=stats,
            )
            
            engine = PremiumStreamEngine(config)
            engine.run(actual_source, record if record not in ("auto", "none") else None)
            
        except Exception as inner_e:
            console.print(f"[bold red]Error:[/bold red] {inner_e}")
            _stream_legacy(actual_source, fps or 15, 1, "standard", "sobel", color != "none")
            
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Stream stopped.[/bold yellow]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)


def _stream_screen_capture(url: Optional[str], duration: Optional[float], mode: str, color: str, record: str, stats: bool):
    """Stream screen capture with optional URL navigation (for Netflix etc.)."""
    try:
        from ..streaming.core.netflix import NetflixCapture, FFmpegCapture, FirefoxController
    except ImportError as e:
        console.print(f"[bold red]Import Error:[/bold red] {e}")
        console.print("Make sure all dependencies are installed.")
        raise typer.Exit(1)
    
    console.print("[bold magenta]═══════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]      🖥️  SCREEN CAPTURE MODE 🖥️         [/bold magenta]")
    console.print("[bold magenta]═══════════════════════════════════════════[/bold magenta]")
    
    # Determine output file
    if record == "auto" or record is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"screen_capture_{timestamp}.mp4"
    elif record == "none":
        output_file = None
    else:
        output_file = record
    
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
            
            # Offer to play the captured file through glyph renderer
            console.print("[yellow]To render as glyph art:[/yellow]")
            console.print(f"[cyan]  glyph-forge stream {output_file}[/cyan]")


def _stream_legacy(source, fps, scale, gradient, algorithm, color):
    """Legacy streaming using glyph_stream.py subprocess."""
    import subprocess
    
    # Find glyph_stream.py
    package_root = Path(__file__).parent.parent.parent.parent
    glyph_stream_path = package_root / "glyph_stream.py"
    
    if not glyph_stream_path.exists():
        glyph_stream_path = Path.cwd() / "glyph_stream.py"
    
    if not glyph_stream_path.exists():
        console.print("[bold red]Error:[/bold red] glyph_stream.py not found.")
        raise typer.Exit(1)
    
    cmd = [sys.executable, str(glyph_stream_path)]
    
    if isinstance(source, int):
        cmd.append("--webcam")
    else:
        cmd.append(str(source))
    
    cmd.extend(["--fps", str(int(fps or 15))])
    cmd.extend(["--scale", str(scale)])
    cmd.extend(["--gradient-set", gradient])
    cmd.extend(["--algorithm", algorithm])
    
    if not color:
        cmd.append("--no-color")
    
    console.print(f"[bold cyan]Launching legacy glyph stream...[/bold cyan]")
    console.print(f"[dim]Command: {' '.join(cmd)}[/dim]")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Stream stopped.[/bold yellow]")
    except FileNotFoundError:
        console.print("[bold red]Error:[/bold red] glyph_stream.py not found.")

@eidosian()
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
    table.add_row("list-commands", "Display this command list")
    
    # Add bannerize subcommands
    table.add_section()
    table.add_row("bannerize", "Generate stylized text banners")
    
    # Add imagize subcommands (the replacement for Glyphfy)
    table.add_section()
    table.add_row("imagize", "Transform images into Glyph art masterpieces")
    
    console.print(table)

@eidosian()
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

@eidosian()
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
    console.print("\nType [cyan]glyph-forge --help[/cyan] for more information\n")

@eidosian()
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

@eidosian()
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
