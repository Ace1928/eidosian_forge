import unittest
from pathlib import Path

from doc_forge import DocForge


class TestDocForge(unittest.TestCase):
    def setUp(self):
        self.forge = DocForge()

    def test_generate_readme(self):
        info = {"name": "TestProj", "description": "A test project", "features": ["Feature A", "Feature B"]}
        readme = self.forge.generate_readme(info)
        self.assertIn("# 🔮 TestProj", readme)
        self.assertIn("- Feature A", readme)

    def test_extract_and_generate(self):
        # Create a temporary source file
        temp_dir = Path("temp_src")
        temp_dir.mkdir(exist_ok=True)
        (temp_dir / "test_mod.py").write_text('"""Module doc"""\ndef my_func():\n    """Func doc"""\n    pass')

        try:
            api_ref = self.forge.extract_and_generate_api_docs(temp_dir)
            self.assertIn("Module: `test_mod.py`", api_ref)
            self.assertIn("def my_func", api_ref)
            self.assertIn("Func doc", api_ref)
        finally:
            import shutil

            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
