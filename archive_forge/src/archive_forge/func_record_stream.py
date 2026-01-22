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
def record_stream(self):
    """Yield substream_type, substream from the byte stream."""

    def wrap_and_count(pb, rc, substream):
        """Yield records from stream while showing progress."""
        counter = 0
        if rc:
            if self.current_type != 'revisions' and self.key_count != 0:
                if not rc.is_initialized():
                    rc.setup(self.key_count, self.key_count)
        for record in substream.read():
            if rc:
                if rc.is_initialized() and counter == rc.STEP:
                    rc.increment(counter)
                    pb.update('Estimate', rc.current, rc.max)
                    counter = 0
                if self.current_type == 'revisions':
                    self.key_count += 1
                    if counter == rc.STEP:
                        pb.update('Estimating..', self.key_count)
                        counter = 0
            counter += 1
            yield record
    self.seed_state()
    with ui.ui_factory.nested_progress_bar() as pb:
        rc = self._record_counter
        try:
            while self.first_bytes is not None:
                substream = NetworkRecordStream(self.iter_substream_bytes())
                yield (self.current_type.decode('ascii'), wrap_and_count(pb, rc, substream))
        finally:
            if rc:
                pb.update('Done', rc.max, rc.max)