import unittest
from figlet_forge import Figlet


class FigletForge:
    """Test wrapper for Figlet with simplified API."""
    
    def __init__(self, font: str = "standard"):
        self.figlet = Figlet(font=font)
    
    def generate(self, text: str, style: str = "elegant") -> str:
        """Generate ASCII art banner."""
        art = self.figlet.render_text(text)
        # Add box based on style
        lines = art.strip().split('\n')
        width = max(len(line) for line in lines) + 4
        
        if style == "elegant":
            box_top = "╭" + "─" * (width - 2) + "╮"
            box_bottom = "╰" + "─" * (width - 2) + "╯"
        elif style == "bold":
            box_top = "┏" + "━" * (width - 2) + "┓"
            box_bottom = "┗" + "━" * (width - 2) + "┛"
        elif style == "double":
            box_top = "╔" + "═" * (width - 2) + "╗"
            box_bottom = "╚" + "═" * (width - 2) + "╝"
        else:
            box_top = "+" + "-" * (width - 2) + "+"
            box_bottom = box_top
            
        return box_top + "\n" + art + "\n" + box_bottom
    
    def render_header(self, text: str) -> str:
        """Render a header with box."""
        return self.generate(text.upper(), style="bold")


class TestFigletForge(unittest.TestCase):
    def setUp(self):
        self.forge = FigletForge()

    def test_generate_basic(self):
        banner = self.forge.generate("Eidos")
        # Check box is present
        self.assertIn("╭", banner)  # Default elegant
        self.assertIn("╯", banner)

    def test_generate_styles(self):
        banner_bold = self.forge.generate("Eidos", style="bold")
        self.assertIn("┏", banner_bold)
        
        banner_double = self.forge.generate("Eidos", style="double")
        self.assertIn("╔", banner_double)

    def test_header_rendering(self):
        header = self.forge.render_header("title")
        self.assertIn("┏", header)

if __name__ == "__main__":
    unittest.main()
