import bz2
import itertools
import os
import queue
import sys
import tempfile
import threading
import zlib
import fastbencode as bencode
from ... import errors, estimate_compressed_size, osutils
from ... import revision as _mod_revision
from ... import trace, ui
from ...repository import _strip_NULL_ghosts, network_format_registry
from .. import inventory as _mod_inventory
from .. import inventory_delta, pack, vf_search
from ..bzrdir import BzrDir
from ..versionedfile import (ChunkedContentFactory, NetworkRecordStream,
from .request import (FailedSmartServerResponse, SmartServerRequest,
def iter_substream_bytes(self):
    if self.first_bytes is not None:
        yield self.first_bytes
        self.first_bytes = None
    for record in self.iter_pack_records:
        record_names, record_bytes = record
        record_name, = record_names
        substream_type = record_name[0]
        if substream_type != self.current_type:
            self.current_type = substream_type
            self.first_bytes = record_bytes
            return
        yield record_bytes