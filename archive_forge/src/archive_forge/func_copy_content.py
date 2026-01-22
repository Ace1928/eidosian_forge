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
def copy_content(self, revision_id=None):
    """See InterRepository.copy_content()."""
    with self.lock_write():
        try:
            self.target.set_make_working_trees(self.source.make_working_trees())
        except (errors.RepositoryUpgradeRequired, NotImplementedError):
            pass
        if self.source._transport.listable():
            with ui.ui_factory.nested_progress_bar() as pb:
                self.target.texts.insert_record_stream(self.source.texts.get_record_stream(self.source.texts.keys(), 'topological', False))
                pb.update('Copying inventory', 0, 1)
                self.target.inventories.insert_record_stream(self.source.inventories.get_record_stream(self.source.inventories.keys(), 'topological', False))
                self.target.signatures.insert_record_stream(self.source.signatures.get_record_stream(self.source.signatures.keys(), 'unordered', True))
                self.target.revisions.insert_record_stream(self.source.revisions.get_record_stream(self.source.revisions.keys(), 'topological', True))
        else:
            self.target.fetch(self.source, revision_id=revision_id)