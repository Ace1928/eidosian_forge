import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List
import pytest
from jupyter_lsp import LanguageServerManager
from ..virtual_documents_shadow import (
@pytest.fixture
def forbidden_shadow_path(tmpdir):
    path = Path(tmpdir) / 'no_permission_dir'
    path.mkdir()
    path.chmod(0)
    yield path
    path.chmod(493)