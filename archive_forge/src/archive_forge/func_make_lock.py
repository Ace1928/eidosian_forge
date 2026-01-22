from io import BytesIO
from ... import errors, lockable_files
from ...bzr.bzrdir import BzrDir, BzrDirFormat, BzrDirMetaFormat1
from ...controldir import (ControlDir, Converter, MustHaveWorkingTree,
from ...i18n import gettext
from ...lazy_import import lazy_import
from ...transport import NoSuchFile, get_transport, local
import os
from breezy import (
from breezy.bzr import (
from breezy.plugins.weave_fmt.store.versioned import VersionedFileStore
from breezy.transactions import WriteTransaction
from breezy.plugins.weave_fmt import xml4
def make_lock(self, name):
    """Make a lock for the new control dir name."""
    self.step(gettext('Make %s lock') % name)
    ld = lockdir.LockDir(self.controldir.transport, '%s/lock' % name, file_modebits=self.file_mode, dir_modebits=self.dir_mode)
    ld.create()