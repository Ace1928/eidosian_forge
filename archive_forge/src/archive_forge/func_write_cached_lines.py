import re
from io import BytesIO
from .lazy_import import lazy_import
from fnmatch import fnmatch
from breezy._termcolor import color_string, FG
from breezy import (
from . import controldir, errors, osutils
from . import revision as _mod_revision
from . import trace
from .revisionspec import RevisionSpec, RevisionSpec_revid, RevisionSpec_revno
def write_cached_lines(self, cache_id, revno):
    """Write cached results out again for new revision"""
    cached_path, cached_matches = self.cache[cache_id]
    start = self._format_initial % {'path': cached_path, 'revno': revno}
    write = self.outf.write
    for end in cached_matches:
        write(start + end)