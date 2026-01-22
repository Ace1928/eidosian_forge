import errno
import os
from stat import S_ISDIR
from wandb_watchdog.utils import stat as default_stat
@property
def files_moved(self):
    """
        List of files that were moved.

        Each event is a two-tuple the first item of which is the path
        that has been renamed to the second item in the tuple.
        """
    return self._files_moved