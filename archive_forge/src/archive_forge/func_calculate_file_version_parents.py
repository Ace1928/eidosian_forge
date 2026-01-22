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
def calculate_file_version_parents(self, text_key):
    """Calculate the correct parents for a file version according to
        the inventories.
        """
    parent_keys = self.text_index[text_key]
    if parent_keys == [_mod_revision.NULL_REVISION]:
        return ()
    return tuple(parent_keys)