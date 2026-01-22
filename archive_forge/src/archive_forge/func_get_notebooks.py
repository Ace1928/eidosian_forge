import logging
import pathlib
import sys
from textwrap import dedent
import nbformat
from jupyter_core.application import JupyterApp
from traitlets import Bool, Integer, List, Unicode, default
from traitlets.config import catch_config_error
from nbclient import __version__
from .client import NotebookClient
def get_notebooks(self):
    """Get the notebooks for the app."""
    if self.extra_args:
        notebooks = self.extra_args
    else:
        notebooks = self.notebooks
    return notebooks