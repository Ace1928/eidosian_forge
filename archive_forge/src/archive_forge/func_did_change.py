import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List
import pytest
from jupyter_lsp import LanguageServerManager
from ..virtual_documents_shadow import (
def did_change(uri, changes: List):
    return {'method': 'textDocument/didChange', 'params': {'textDocument': {'uri': uri}, 'contentChanges': changes}}