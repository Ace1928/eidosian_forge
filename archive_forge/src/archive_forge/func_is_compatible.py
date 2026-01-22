import gzip
import os
from io import BytesIO
from ...lazy_import import lazy_import
import itertools
from breezy import (
from breezy.bzr import (
from ... import debug, errors, lockable_files, lockdir, osutils, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import tuned_gzip, versionedfile, weave, weavefile
from ...bzr.repository import RepositoryFormatMetaDir
from ...bzr.versionedfile import (AbsentContentFactory, FulltextContentFactory,
from ...bzr.vf_repository import (InterSameDataRepository,
from ...repository import InterRepository
from . import bzrdir as weave_bzrdir
from .store.text import TextStore
@staticmethod
def is_compatible(source, target):
    """Be compatible with known Weave formats.

        We don't test for the stores being of specific types because that
        could lead to confusing results, and there is no need to be
        overly general.
        """
    try:
        return isinstance(source._format, (RepositoryFormat5, RepositoryFormat6, RepositoryFormat7)) and isinstance(target._format, (RepositoryFormat5, RepositoryFormat6, RepositoryFormat7))
    except AttributeError:
        return False