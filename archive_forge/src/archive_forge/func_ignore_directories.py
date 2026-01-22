import os.path
import logging
import re
from .patterns import match_any_paths
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import unicode_paths
@property
def ignore_directories(self):
    """
        (Read-only)
        ``True`` if directories should be ignored; ``False`` otherwise.
        """
    return self._ignore_directories