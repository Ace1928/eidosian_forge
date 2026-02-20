import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import shutil
from repo_forge.generators.scripts import create_script_files


class TestScriptGenerator(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.mkdtemp(prefix="repo_forge_test_")
        self.base_path = Path(self._tmp_dir) / "test_repo"

    def tearDown(self):
        shutil.rmtree(self._tmp_dir, ignore_errors=True)

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
