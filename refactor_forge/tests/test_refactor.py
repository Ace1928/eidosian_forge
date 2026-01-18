import unittest
import ast
from eidosian_forge.refactor_forge import RefactorForge

class TestRefactorForge(unittest.TestCase):
    def setUp(self):
        self.forge = RefactorForge()

    def test_rename_variables(self):
        source = "x = 1\ny = x + 1"
        rename_map = {"x": "my_var"}
        new_source = self.forge.transform(source, rename_map=rename_map)
        self.assertIn("my_var = 1", new_source)
        self.assertIn("y = my_var + 1", new_source)

    def test_remove_docstrings(self):
        source = 'def f():\n    """Docstring"""\n    return 1'
        new_source = self.forge.remove_docstrings(source)
        self.assertNotIn("Docstring", new_source)
        self.assertIn("return 1", new_source)

    def test_extract_pattern(self):
        source = "def a(): pass\ndef b(): pass\nx = 1"
        # Extract FunctionDef
        functions = self.forge.extract_pattern(source, ast.FunctionDef)
        self.assertEqual(len(functions), 2)
        self.assertIn("def a():", functions[0])

if __name__ == "__main__":
    unittest.main()

