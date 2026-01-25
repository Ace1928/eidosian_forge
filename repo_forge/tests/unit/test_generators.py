import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from repo_forge.generators.scripts import create_script_files


class TestScriptGenerator(unittest.TestCase):
    def setUp(self):
        self.base_path = Path("/tmp/test_repo")

    @patch('repo_forge.generators.scripts.write_file')
    @patch('repo_forge.generators.scripts.make_executable')
    @patch('repo_forge.generators.scripts.render_template')
    def test_create_script_files(self, mock_render, mock_make_exec, mock_write):
        # Setup mock render_template
        mock_render.return_value = "rendered content"
        
        # Run generator
        result = create_script_files(self.base_path, languages=["python"])
        
        # Verify
        self.assertTrue(result["success"])
        self.assertGreaterEqual(result["count"], 0)


if __name__ == "__main__":
    unittest.main()
