import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List
import pytest
from jupyter_lsp import LanguageServerManager
from ..virtual_documents_shadow import (
def run_shadow(message):
    return shadow('client', message, 'python-lsp-server', manager)