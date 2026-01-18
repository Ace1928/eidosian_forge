import unittest
from eidosian_forge.figlet_forge import FigletForge

class TestFigletForge(unittest.TestCase):
    def setUp(self):
        self.forge = FigletForge()

    def test_generate_basic(self):
        banner = self.forge.generate("Eidos")
        self.assertIn("Eidos", banner)
        self.assertIn("╭", banner) # Default elegant
        self.assertIn("╯", banner)

    def test_generate_styles(self):
        banner_bold = self.forge.generate("Eidos", style="bold")
        self.assertIn("┏", banner_bold)
        
        banner_double = self.forge.generate("Eidos", style="double")
        self.assertIn("╔", banner_double)

    def test_header_rendering(self):
        header = self.forge.render_header("title")
        self.assertIn("TITLE", header) # Should be uppercased
        self.assertIn("┏", header)

if __name__ == "__main__":
    unittest.main()
