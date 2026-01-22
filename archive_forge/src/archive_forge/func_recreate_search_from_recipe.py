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
def recreate_search_from_recipe(self, repository, lines, discard_excess=False):
    """Recreate a specific revision search (vs a from-tip search).

        :param discard_excess: If True, and the search refers to data we don't
            have, just silently accept that fact - the verb calling
            recreate_search trusts that clients will look for missing things
            they expected and get it from elsewhere.
        """
    start_keys = set(lines[0].split(b' '))
    exclude_keys = set(lines[1].split(b' '))
    revision_count = int(lines[2].decode('ascii'))
    with repository.lock_read():
        search = repository.get_graph()._make_breadth_first_searcher(start_keys)
        while True:
            try:
                next_revs = next(search)
            except StopIteration:
                break
            search.stop_searching_any(exclude_keys.intersection(next_revs))
        started_keys, excludes, included_keys = search.get_state()
        if not discard_excess and len(included_keys) != revision_count:
            return (None, FailedSmartServerResponse((b'NoSuchRevision',)))
        search_result = vf_search.SearchResult(started_keys, excludes, len(included_keys), included_keys)
        return (search_result, None)