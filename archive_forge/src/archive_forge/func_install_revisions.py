import bz2
import re
from io import BytesIO
import fastbencode as bencode
from .... import errors, iterablefile, lru_cache, multiparent, osutils
from .... import repository as _mod_repository
from .... import revision as _mod_revision
from .... import trace, ui
from ....i18n import ngettext
from ... import pack, serializer
from ... import versionedfile as _mod_versionedfile
from .. import bundle_data
from .. import serializer as bundle_serializer
def install_revisions(self, repository, stream_input=True):
    """Install this bundle's revisions into the specified repository

        :param target_repo: The repository to install into
        :param stream_input: If True, will stream input rather than reading it
            all into memory at once.  Reading it into memory all at once is
            (currently) faster.
        """
    with repository.lock_write():
        ri = RevisionInstaller(self.get_bundle_reader(stream_input), self._serializer, repository)
        return ri.install()