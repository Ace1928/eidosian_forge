import logging
import copy
from ..initializer import Uniform
from .base_module import BaseModule
def install_monitor(self, mon):
    """Installs monitor on all executors."""
    assert self.binded
    for module in self._modules:
        module.install_monitor(mon)