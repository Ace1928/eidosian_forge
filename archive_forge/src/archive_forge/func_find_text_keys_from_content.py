from .. import errors
from .. import transport as _mod_transport
from ..lazy_import import lazy_import
import time
from breezy import (
from breezy.bzr import (
from breezy.bzr.knit import (
from ..bzr import btree_index
from ..bzr.index import (CombinedGraphIndex, GraphIndex,
from ..bzr.vf_repository import StreamSource
from .knitrepo import KnitRepository
from .pack_repo import (NewPack, PackCommitBuilder, Packer, PackRepository,
def find_text_keys_from_content(record):
    if record.storage_kind not in ('knit-delta-gz', 'knit-ft-gz'):
        raise ValueError('Unknown content storage kind for inventory text: %s' % (record.storage_kind,))
    raw_data = record._raw_record
    revision_id = record.key[-1]
    content, _ = knit._parse_record(revision_id, raw_data)
    if record.storage_kind == 'knit-delta-gz':
        line_iterator = factory.get_linedelta_content(content)
    elif record.storage_kind == 'knit-ft-gz':
        line_iterator = factory.get_fulltext_content(content)
    content_text_keys.update(find_text_keys([(line, revision_id) for line in line_iterator]))