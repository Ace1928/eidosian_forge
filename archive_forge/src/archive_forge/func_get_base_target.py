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
@staticmethod
def get_base_target(revision_ids, forced_bases, repository):
    """Determine the base and target from old-style revision ids"""
    if len(revision_ids) == 0:
        return (None, None)
    target = revision_ids[0]
    base = forced_bases.get(target)
    if base is None:
        parents = repository.get_revision(target).parent_ids
        if len(parents) == 0:
            base = _mod_revision.NULL_REVISION
        else:
            base = parents[0]
    return (base, target)