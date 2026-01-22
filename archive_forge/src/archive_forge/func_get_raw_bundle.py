import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
def get_raw_bundle(self):
    if self.bundle is None:
        return None
    else:
        return base64.b64decode(self.bundle)