from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def rename_remote(self, old_relpath, new_relpath):
    """Rename a remote file or directory taking care of collisions.

        To avoid collisions during bulk renames, each renamed target is
        temporarily assigned a unique name. When all renames have been done,
        each target get its proper name.
        """
    import os
    import random
    import time
    stamp = '.tmp.%.9f.%d.%d' % (time.time(), os.getpid(), random.randint(0, 2147483647))
    if not self.quiet:
        self.outf.write('Renaming {} to {}\n'.format(old_relpath, new_relpath))
    self._up_rename(old_relpath, stamp)
    self._pending_renames.append((stamp, new_relpath))