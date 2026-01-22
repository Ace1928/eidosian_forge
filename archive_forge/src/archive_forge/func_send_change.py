import logging
from pathlib import Path
from types import SimpleNamespace
from typing import List
import pytest
from jupyter_lsp import LanguageServerManager
from ..virtual_documents_shadow import (
def send_change():
    message = did_open(file_uri, 'content')
    return shadow('client', message, 'python-lsp-server', manager)