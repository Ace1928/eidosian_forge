from io import BytesIO
import fastbencode as bencode
from .. import lazy_import
from breezy.bzr import (
from .. import cache_utf8, errors
from .. import revision as _mod_revision
from . import serializer
def read_inventory_from_lines(self, xml_lines, revision_id=None, entry_cache=None, return_from_cache=False):
    """Read xml_string into an inventory object.

        :param xml_string: The xml to read.
        :param revision_id: If not-None, the expected revision id of the
            inventory.
        :param entry_cache: An optional cache of InventoryEntry objects. If
            supplied we will look up entries via (file_id, revision_id) which
            should map to a valid InventoryEntry (File/Directory/etc) object.
        :param return_from_cache: Return entries directly from the cache,
            rather than copying them first. This is only safe if the caller
            promises not to mutate the returned inventory entries, but it can
            make some operations significantly faster.
        """
    try:
        return self._unpack_inventory(xml_serializer.fromstringlist(xml_lines), revision_id, entry_cache=entry_cache, return_from_cache=return_from_cache)
    except xml_serializer.ParseError as e:
        raise serializer.UnexpectedInventoryFormat(e)