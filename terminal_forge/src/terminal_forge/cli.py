from eidosian_core import eidosian
#!/usr/bin/env python3
"""
Terminal Forge CLI - Terminal styling and display tools.

Provides commands for colors, themes, banners, and ASCII art.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add lib to path for StandardCLI
LIB_PATH = Path(__file__).parent.parent.parent.parent / "lib"
sys.path.insert(0, str(LIB_PATH))

from cli import StandardCLI, CommandResult, Colors as CLIColors


class TerminalForgeCLI(StandardCLI):
    """CLI for Terminal Forge - terminal styling and display tools."""
    
    name = "terminal_forge"
    description = "Terminal styling, colors, themes, and ASCII art"
    version = "1.0.0"
    
    def __init__(self):
        super().__init__()
        self._forge = None
    
    @property
    def forge(self):
        """Lazy load terminal_forge modules."""
        if self._forge is None:
            self._forge = {}
            try:
                from terminal_forge import colors, themes, banner, ascii_art, layout
                self._forge['colors'] = colors
                self._forge['themes'] = themes
                self._forge['banner'] = banner
                self._forge['ascii_art'] = ascii_art
                self._forge['layout'] = layout
            except ImportError as e:
                self._forge['error'] = str(e)
        return self._forge
    
    @eidosian()
    def register_commands(self, subparsers):
        """Register terminal forge commands."""
        
        # Colors command
        colors_parser = subparsers.add_parser(
            "colors",
            help="List available colors"
        )
        colors_parser.add_argument(
            "--show",
            action="store_true",
            help="Show color samples"
        )
        colors_parser.set_defaults(func=self._cmd_colors)
        
        # Themes command
        themes_parser = subparsers.add_parser(
            "themes",
            help="List available themes"
        )
        themes_parser.add_argument(
            "--apply",
            metavar="NAME",
            help="Apply a theme"
        )
        themes_parser.set_defaults(func=self._cmd_themes)
        
        # Banner command
        banner_parser = subparsers.add_parser(
            "banner",
            help="Create a styled banner"
        )
        banner_parser.add_argument(
            "text",
            nargs="*",
            help="Text to display in banner"
        )
        banner_parser.add_argument(
            "--title",
            help="Banner title"
        )
        banner_parser.add_argument(
            "--border",
            choices=["single", "double", "rounded", "bold", "none"],
            default="single",
            help="Border style"
        )
        banner_parser.add_argument(
            "--color",
            help="Border color"
        )
        banner_parser.set_defaults(func=self._cmd_banner)
        
        # ASCII art command
        ascii_parser = subparsers.add_parser(
            "ascii",
            help="Create ASCII art from image"
        )
        ascii_parser.add_argument(
            "source",
            nargs="?",
            help="Image file or URL"
        )
        ascii_parser.add_argument(
            "--width",
            type=int,
            default=80,
            help="Output width in characters"
        )
        ascii_parser.add_argument(
            "--charset",
            choices=["standard", "blocks", "braille", "detailed"],
            default="standard",
            help="Character set to use"
        )
        ascii_parser.set_defaults(func=self._cmd_ascii)
        
        # Layout command
        layout_parser = subparsers.add_parser(
            "layout",
            help="Show layout capabilities"
        )
        layout_parser.set_defaults(func=self._cmd_layout)
    
    @eidosian()
    def cmd_status(self, args) -> CommandResult:
        """Show terminal forge status."""
        try:
            forge = self.forge
            
            if 'error' in forge:
                return CommandResult(False, f"Import error: {forge['error']}")
            
            colors_mod = forge.get('colors')
            themes_mod = forge.get('themes')
            
            # Get available colors
            color_count = len([c for c in dir(colors_mod.Color) 
                             if not c.startswith('_') and c.isupper()])
            
            # Get available themes
            theme_count = len(themes_mod.Theme.list_themes()) if hasattr(themes_mod.Theme, 'list_themes') else 0
            
            data = {
                "status": "operational",
                "colors": color_count,
                "themes": theme_count,
                "modules": ["colors", "themes", "banner", "ascii_art", "layout"],
            }
            
            return CommandResult(True, "Terminal forge operational", data)
        except Exception as e:
            return CommandResult(False, f"Error: {e}")
    
    def _cmd_colors(self, args) -> int:
        """List available colors."""
        forge = self.forge
        if 'error' in forge:
            print(f"{CLIColors.RED}Error:{CLIColors.RESET} {forge['error']}")
            return 1
        
        Color = forge['colors'].Color
        colors = [c for c in dir(Color) if not c.startswith('_') and c.isupper()]
        
        if args.json:
            import json
            print(json.dumps({"colors": colors}))
        else:
            print(f"{CLIColors.BOLD}Available Colors ({len(colors)}):{CLIColors.RESET}")
            print()
            
            if args.show:
                for color in colors[:20]:  # Limit to 20
                    code = getattr(Color, color, '')
                    print(f"  {code}{color:20}{Color.RESET if hasattr(Color, 'RESET') else ''}")
            else:
                # Grid display
                for i in range(0, len(colors), 4):
                    row = colors[i:i+4]
                    print("  " + "  ".join(f"{c:15}" for c in row))
        
        return 0
    
    def _cmd_themes(self, args) -> int:
        """List or apply themes."""
        forge = self.forge
        if 'error' in forge:
            print(f"{CLIColors.RED}Error:{CLIColors.RESET} {forge['error']}")
            return 1
        
        Theme = forge['themes'].Theme
        
        if args.apply:
            try:
                Theme.apply(args.apply)
                print(f"{CLIColors.GREEN}✓{CLIColors.RESET} Applied theme: {args.apply}")
            except Exception as e:
                print(f"{CLIColors.RED}Error:{CLIColors.RESET} {e}")
                return 1
        else:
            themes = Theme.list_themes() if hasattr(Theme, 'list_themes') else []
            
            if args.json:
                import json
                print(json.dumps({"themes": themes}))
            else:
                print(f"{CLIColors.BOLD}Available Themes:{CLIColors.RESET}")
                for theme in themes:
                    print(f"  - {theme}")
                if not themes:
                    print("  (No themes registered)")
        
        return 0
    
    def _cmd_banner(self, args) -> int:
        """Create a styled banner."""
        forge = self.forge
        if 'error' in forge:
            print(f"{CLIColors.RED}Error:{CLIColors.RESET} {forge['error']}")
            return 1
        
        Banner = forge['banner'].Banner
        
        text = " ".join(args.text) if args.text else "EIDOS"
        
        try:
            b = Banner()
            if args.title:
                b.add_line(args.title)
                b.add_separator()
            b.add_line(text)
            
            if args.border and args.border != "none":
                b.set_border(args.border)
            
            b.display()
        except Exception as e:
            print(f"{CLIColors.RED}Error:{CLIColors.RESET} {e}")
            return 1
        
        return 0
    
    def _cmd_ascii(self, args) -> int:
        """Create ASCII art from image."""
        forge = self.forge
        if 'error' in forge:
            print(f"{CLIColors.RED}Error:{CLIColors.RESET} {forge['error']}")
            return 1
        
        if not args.source:
            print(f"{CLIColors.YELLOW}Usage:{CLIColors.RESET} terminal-forge ascii <image_file>")
            print()
            print("Options:")
            print("  --width N       Output width (default: 80)")
            print("  --charset TYPE  Character set: standard, blocks, braille, detailed")
            return 0
        
        AsciiArtBuilder = forge['ascii_art'].AsciiArtBuilder
        
        try:
            builder = AsciiArtBuilder.from_source(args.source)
            builder.width(args.width)
            # Apply charset if supported
            result = builder.print()
        except Exception as e:
            print(f"{CLIColors.RED}Error:{CLIColors.RESET} {e}")
            return 1
        
        return 0
    
    def _cmd_layout(self, args) -> int:
        """Show layout capabilities."""
        forge = self.forge
        if 'error' in forge:
            print(f"{CLIColors.RED}Error:{CLIColors.RESET} {forge['error']}")
            return 1
        
        layout = forge['layout']
        
        # Get available classes
        classes = [c for c in dir(layout) if not c.startswith('_') and c[0].isupper()]
        
        if args.json:
            import json
            print(json.dumps({"layout_classes": classes}))
        else:
            print(f"{CLIColors.BOLD}Layout Capabilities:{CLIColors.RESET}")
            for cls in classes[:10]:
                print(f"  - {cls}")
        
        return 0


@eidosian()
def main():
    """Entry point for terminal-forge CLI."""
    cli = TerminalForgeCLI()
    return cli.run()


if __name__ == "__main__":
    sys.exit(main())
