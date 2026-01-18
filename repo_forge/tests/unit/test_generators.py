import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from repo_forge.generators.scripts import create_script_files

class TestScriptGenerator(unittest.TestCase):
    def setUp(self):
        self.base_path = Path("/tmp/test_repo")

    @patch('repo_forge.generators.scripts.write_file')
    @patch('repo_forge.generators.scripts.make_executable')
    @patch('repo_forge.generators.scripts.TemplateManager')
    def test_create_script_files(self, mock_tm_class, mock_make_exec, mock_write):
        # Setup mock TemplateManager
        mock_tm = mock_tm_class.return_value
        mock_tm.render.return_value = "rendered content"
        
        # Run generator
        result = create_script_files(self.base_path, languages=["python"])
        
        # Verify
        self.assertTrue(result["success"])
        self.assertGreater(result["count"], 0)
        
        # Verify README was rendered and written
        mock_tm.render.assert_any_call("scripts_readme", {})
        
        # Verify stats script for python was rendered
        mock_tm.render.assert_any_call("script_project_stats", unittest.mock.ANY)

if __name__ == "__main__":
    unittest.main()
