from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def delete_remote_dir(self, relpath):
    if not self.quiet:
        self.outf.write('Deleting %s\n' % relpath)
    self._up_rmdir(relpath)