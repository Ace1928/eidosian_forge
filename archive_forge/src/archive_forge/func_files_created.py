import errno
import os
from stat import S_ISDIR
from wandb_watchdog.utils import stat as default_stat
@property
def files_created(self):
    """List of files that were created."""
    return self._files_created