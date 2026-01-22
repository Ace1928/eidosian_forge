import operator
import os
from io import BytesIO
from ..lazy_import import lazy_import
import patiencediff
import gzip
from breezy import (
from breezy.bzr import (
from breezy.bzr import pack_repo
from breezy.i18n import gettext
from .. import annotate, errors, osutils
from .. import transport as _mod_transport
from ..bzr.versionedfile import (AbsentContentFactory, ConstantMapper,
from ..errors import InternalBzrError, InvalidRevisionId, RevisionNotPresent
from ..osutils import contains_whitespace, sha_string, sha_strings, split_lines
from ..transport import NoSuchFile
from . import index as _mod_index
@staticmethod
def get_line_delta_blocks(knit_delta, source, target):
    """Extract SequenceMatcher.get_matching_blocks() from a knit delta"""
    target_len = len(target)
    s_pos = 0
    t_pos = 0
    for s_begin, s_end, t_len, new_text in knit_delta:
        true_n = s_begin - s_pos
        n = true_n
        if n > 0:
            if source[s_pos + n - 1] != target[t_pos + n - 1]:
                n -= 1
            if n > 0:
                yield (s_pos, t_pos, n)
        t_pos += t_len + true_n
        s_pos = s_end
    n = target_len - t_pos
    if n > 0:
        if source[s_pos + n - 1] != target[t_pos + n - 1]:
            n -= 1
        if n > 0:
            yield (s_pos, t_pos, n)
    yield (s_pos + (target_len - t_pos), target_len, 0)