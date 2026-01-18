import unittest
import ast
from eidosian_forge.code_forge import CodeForge

class TestCodeForge(unittest.TestCase):
    def setUp(self):
        self.forge = CodeForge()

    def test_analysis(self):
        source = "def hello(): pass\nclass World: pass"
        analysis = self.forge.analyze_source(source)
        
        self.assertTrue(analysis["valid"])
        self.assertIn("hello", analysis["functions"])
        self.assertIn("World", analysis["classes"])

    def test_invalid_source(self):
        source = "def hello(:" # Syntax error
        analysis = self.forge.analyze_source(source)
        self.assertFalse(analysis["valid"])
        self.assertIn("error", analysis)

    def test_pattern_finding(self):
        source = "x = 1\ny = 2\ndef f(): pass"
        tree = ast.parse(source)
        assignments = self.forge.find_pattern(tree, ast.Assign)
        self.assertEqual(len(assignments), 2)

if __name__ == "__main__":
    unittest.main()

