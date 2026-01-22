from io import BytesIO
from ..lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import serializer
from breezy.i18n import gettext
from breezy.bzr.testament import Testament
from .. import errors
from ..decorators import only_raises
from ..repository import (CommitBuilder, FetchResult, InterRepository,
from ..trace import mutter, note
from .inventory import ROOT_ID, Inventory, entry_factory
from .inventorytree import InventoryTreeChange
from .repository import MetaDirRepository, RepositoryFormatMetaDir
def insert_stream_without_locking(self, stream, src_format, is_resume=False):
    """Insert a stream's content into the target repository.

        This assumes that you already have a locked repository and an active
        write group.

        :param src_format: a bzr repository format.
        :param is_resume: Passed down to get_missing_parent_inventories to
            indicate if we should be checking for missing texts at the same
            time.

        :return: A set of keys that are missing.
        """
    if not self.target_repo.is_write_locked():
        raise errors.ObjectNotLocked(self)
    if not self.target_repo.is_in_write_group():
        raise errors.BzrError('you must already be in a write group')
    to_serializer = self.target_repo._format._serializer
    src_serializer = src_format._serializer
    new_pack = None
    if to_serializer == src_serializer:
        try:
            new_pack = self.target_repo._pack_collection._new_pack
        except AttributeError:
            pass
        else:
            new_pack.set_write_cache_size(1024 * 1024)
    for substream_type, substream in stream:
        if 'stream' in debug.debug_flags:
            mutter('inserting substream: %s', substream_type)
        if substream_type == 'texts':
            self.target_repo.texts.insert_record_stream(substream)
        elif substream_type == 'inventories':
            if src_serializer == to_serializer:
                self.target_repo.inventories.insert_record_stream(substream)
            else:
                self._extract_and_insert_inventories(substream, src_serializer)
        elif substream_type == 'inventory-deltas':
            self._extract_and_insert_inventory_deltas(substream, src_serializer)
        elif substream_type == 'chk_bytes':
            self.target_repo.chk_bytes.insert_record_stream(substream)
        elif substream_type == 'revisions':
            if src_serializer == to_serializer:
                self.target_repo.revisions.insert_record_stream(substream)
            else:
                self._extract_and_insert_revisions(substream, src_serializer)
        elif substream_type == 'signatures':
            self.target_repo.signatures.insert_record_stream(substream)
        else:
            raise AssertionError('kaboom! {}'.format(substream_type))
    if new_pack is not None:
        new_pack._write_data(b'', flush=True)
    missing_keys = self.target_repo.get_missing_parent_inventories(check_for_missing_texts=is_resume)
    try:
        for prefix, versioned_file in (('texts', self.target_repo.texts), ('inventories', self.target_repo.inventories), ('revisions', self.target_repo.revisions), ('signatures', self.target_repo.signatures), ('chk_bytes', self.target_repo.chk_bytes)):
            if versioned_file is None:
                continue
            missing_keys.update(((prefix,) + key for key in versioned_file.get_missing_compression_parent_keys()))
    except NotImplementedError:
        missing_keys = set()
    return missing_keys