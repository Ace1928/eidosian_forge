from typing import Any
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from traitlets import Bool, Enum, Instance, Unicode
from ._version import __version__
from .config import LabConfig
from .licenses_handler import LicensesManager
def init_licenses_manager(self) -> None:
    """Initialize the license manager."""
    self.licenses_manager = LicensesManager(parent=self)