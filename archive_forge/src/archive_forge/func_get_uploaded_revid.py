from ... import commands, config, errors, lazy_import, option, osutils
import stat
from breezy import (
def get_uploaded_revid(self):
    if self._uploaded_revid is None:
        revid_path = self.branch.get_config_stack().get('upload_revid_location')
        try:
            self._uploaded_revid = self._up_get_bytes(revid_path)
        except transport.NoSuchFile:
            self._uploaded_revid = revision.NULL_REVISION
    return self._uploaded_revid