from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def upload_symlink_robustly(self, relpath, target):
    """Handle uploading symlinks.
        """
    self._force_clear(relpath)
    target = osutils.normpath(osutils.pathjoin(osutils.dirname(relpath), target))
    self.upload_symlink(relpath, target)