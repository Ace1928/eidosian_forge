import unittest
import shutil
from pathlib import Path
from eidosian_forge.file_forge import FileForge

class TestFileForge(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_sandbox")
        self.test_dir.mkdir(exist_ok=True)
        self.forge = FileForge(base_path=self.test_dir)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_hashing_and_duplicates(self):
        (self.test_dir / "orig.txt").write_text("content")
        (self.test_dir / "dup.txt").write_text("content")
        (self.test_dir / "other.txt").write_text("other")
        
        dups = self.forge.find_duplicates(self.test_dir)
        self.assertEqual(len(dups), 1) # One set of duplicates
        self.assertEqual(len(list(dups.values())[0]), 2)

    def test_ensure_structure(self):
        structure = {
            "docs": {
                "readme.md": "Content",
                "api": {}
            },
            "main.py": "print(1)"
        }
        self.forge.ensure_structure(structure)
        self.assertTrue((self.test_dir / "docs" / "readme.md").exists())
        self.assertTrue((self.test_dir / "docs" / "api").is_dir())
        self.assertTrue((self.test_dir / "main.py").exists())

    def test_categorize_files(self):
        (self.test_dir / "test.txt").touch()
        (self.test_dir / "test.py").touch()
        
        categories = self.forge.categorize_files(self.test_dir)
        self.assertIn("txt", categories)
        self.assertIn("py", categories)

if __name__ == "__main__":
    unittest.main()
