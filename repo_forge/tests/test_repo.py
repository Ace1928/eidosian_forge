import pytest
from unittest.mock import patch, MagicMock
from repo_forge.cli import main

def test_repo_forge_help(capsys):
    # Test --help
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    
    captured = capsys.readouterr()
    assert "Eidosian Repo Forge" in captured.out

@patch("repo_forge.cli.create_directory_structure")
@patch("repo_forge.cli.create_documentation_structure")
@patch("repo_forge.cli.create_configuration_files")
@patch("repo_forge.cli.create_script_files")
@patch("repo_forge.cli.create_project_scaffold")
def test_repo_creation(mock_proj, mock_script, mock_config, mock_doc, mock_dir, tmp_path):
    # Mock return values
    mock_dir.return_value = {"success": True, "created_files": []}
    mock_doc.return_value = {"success": True, "created_files": []}
    mock_config.return_value = {"success": True, "created_files": []}
    mock_script.return_value = {"success": True, "created_files": []}
    mock_proj.return_value = {"success": True, "created_files": []}
    
    output_dir = tmp_path / "new_repo"
    args = ["-o", str(tmp_path), "-n", "new_repo"]
    
    ret = main(args)
    assert ret == 0
    
    mock_dir.assert_called()