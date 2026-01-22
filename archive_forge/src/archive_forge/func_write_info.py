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
def write_info(self):
    """Write format info"""
    serializer_format = self.repository.get_serializer_format()
    supports_rich_root = {True: 1, False: 0}[self.repository.supports_rich_root()]
    self.bundle.add_info_record({b'serializer': serializer_format, b'supports_rich_root': supports_rich_root})