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
def do_insert_stream_request(self, repository, resume_tokens):
    tokens = [token.decode('utf-8') for token in resume_tokens.split(b' ') if token]
    self.tokens = tokens
    self.repository = repository
    self.queue = queue.Queue()
    self.insert_thread = threading.Thread(target=self._inserter_thread)
    self.insert_thread.start()