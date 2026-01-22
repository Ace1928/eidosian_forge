import glob
import sys
from os import getenv, path
from typing import Any, Dict, List
from traitlets.config import SingletonConfigurable, Unicode, default
from .provisioner_base import KernelProvisionerBase
def get_provisioner_entries(self) -> Dict[str, str]:
    """
        Returns a dictionary of provisioner entries.

        The key is the provisioner name for its entry point.  The value is the colon-separated
        string of the entry point's module name and object name.
        """
    entries = {}
    for name, ep in self.provisioners.items():
        entries[name] = ep.value
    return entries