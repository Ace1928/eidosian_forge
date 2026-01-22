import bz2
import itertools
import os
import queue
import sys
import tempfile
import threading
import zlib
import fastbencode as bencode
from ... import errors, estimate_compressed_size, osutils
from ... import revision as _mod_revision
from ... import trace, ui
from ...repository import _strip_NULL_ghosts, network_format_registry
from .. import inventory as _mod_inventory
from .. import inventory_delta, pack, vf_search
from ..bzrdir import BzrDir
from ..versionedfile import (ChunkedContentFactory, NetworkRecordStream,
from .request import (FailedSmartServerResponse, SmartServerRequest,
def recreate_search(self, repository, search_bytes, discard_excess=False):
    """Recreate a search from its serialised form.

        :param discard_excess: If True, and the search refers to data we don't
            have, just silently accept that fact - the verb calling
            recreate_search trusts that clients will look for missing things
            they expected and get it from elsewhere.
        """
    if search_bytes == b'everything':
        return (vf_search.EverythingResult(repository), None)
    lines = search_bytes.split(b'\n')
    if lines[0] == b'ancestry-of':
        heads = lines[1:]
        search_result = vf_search.PendingAncestryResult(heads, repository)
        return (search_result, None)
    elif lines[0] == b'search':
        return self.recreate_search_from_recipe(repository, lines[1:], discard_excess=discard_excess)
    else:
        return (None, FailedSmartServerResponse((b'BadSearch',)))