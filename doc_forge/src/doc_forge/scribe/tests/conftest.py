import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock
from doc_forge.scribe.config import ScribeConfig

@pytest.fixture
def temp_forge_root():
    """Create a temporary forge root for testing."""
    temp_dir = tempfile.mkdtemp()
    root = Path(temp_dir)
    
    # Create necessary subdirs
    (root / "doc_forge" / "runtime").mkdir(parents=True)
    (root / "models").mkdir(parents=True)
    (root / "llama.cpp" / "build" / "bin").mkdir(parents=True)
    
    # Create dummy model and binary
    (root / "models" / "dummy.gguf").touch()
    (root / "llama.cpp" / "build" / "bin" / "llama-server").touch(mode=0o755)
    
    yield root
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_config(temp_forge_root):
    return ScribeConfig.from_env(
        forge_root=temp_forge_root,
        host="127.0.0.1",
        port=8000,
        dry_run=True
    )

@pytest.fixture
def mock_llm_response():
    """Mock successful LLM response."""
    mock = MagicMock()
    mock.status_code = 200
    mock.json.return_value = {"content": "# File Overview\n\nGenerated content."}
    return mock
