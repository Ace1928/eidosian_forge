import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List
import pytest
from jupyter_lsp import LanguageServerManager
from ..virtual_documents_shadow import (
def did_save_with_text(uri, text):
    return {'method': 'textDocument/didSave', 'params': {'textDocument': {'uri': uri, 'text': text}}}