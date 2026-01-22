import warnings
import numpy
from sacred.dependencies import get_digest
from sacred.observers import RunObserver
import wandb
def resource_event(self, filename):
    """TODO: Maintain resources list."""
    if filename not in self.resources:
        md5 = get_digest(filename)
        self.resources[filename] = md5