import collections
import os
import socket
import sys
import time
from functools import partial
from typing import Dict, Iterable, List, Optional, Set, Tuple
import socketserver
import zlib
from dulwich import log_utils
from .archive import tar_stream
from .errors import (
from .object_store import peel_sha
from .objects import Commit, ObjectID, valid_hexsha
from .pack import ObjectContainer, PackedObjectContainer, write_pack_from_container
from .protocol import (
from .refs import PEELED_TAG_SUFFIX, RefsContainer, write_info_refs
from .repo import BaseRepo, Repo
@classmethod
def innocuous_capabilities(cls) -> Iterable[bytes]:
    return [CAPABILITY_INCLUDE_TAG, CAPABILITY_THIN_PACK, CAPABILITY_NO_PROGRESS, CAPABILITY_OFS_DELTA, capability_agent()]