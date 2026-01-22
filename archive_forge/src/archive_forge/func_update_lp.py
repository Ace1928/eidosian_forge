import base64
import re
from urllib.parse import urlparse, urlunparse
from ... import bedding, branch, errors, osutils, trace, transport
from ...i18n import gettext
from launchpadlib.credentials import (AccessToken, Credentials,
from launchpadlib.launchpad import Launchpad
def update_lp(self):
    """Update the Launchpad copy of this branch."""
    if not self._check_update:
        return
    with self.bzr.lock_read():
        if self.lp.last_scanned_id is not None:
            if self.bzr.last_revision() == self.lp.last_scanned_id:
                trace.note(gettext('%s is already up-to-date.') % self.lp.bzr_identity)
                return
            graph = self.bzr.repository.get_graph()
            if not graph.is_ancestor(osutils.safe_utf8(self.lp.last_scanned_id), self.bzr.last_revision()):
                raise errors.DivergedBranches(self.bzr, self.push_bzr)
            trace.note(gettext('Pushing to %s') % self.lp.bzr_identity)
        self.bzr.push(self.push_bzr)