from __future__ import absolute_import
from .. import (
from ..helpers import (
import stat
def pre_process(self):
    self.cmd_counts = {}
    for cmd in commands.COMMAND_NAMES:
        self.cmd_counts[cmd] = 0
    self.file_cmd_counts = {}
    for fc in commands.FILE_COMMAND_NAMES:
        self.file_cmd_counts[fc] = 0
    self.parent_counts = {}
    self.max_parent_count = 0
    self.committers = set()
    self.separate_authors_found = False
    self.symlinks_found = False
    self.executables_found = False
    self.sha_blob_references = False
    self.lightweight_tags = 0
    self.blobs = {}
    for usage in ['new', 'used', 'unknown', 'unmarked']:
        self.blobs[usage] = set()
    self.blob_ref_counts = {}
    self.reftracker = reftracker.RefTracker()
    self.merges = {}
    self.rename_old_paths = {}
    self.copy_source_paths = {}