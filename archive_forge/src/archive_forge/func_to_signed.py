import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
def to_signed(self, branch):
    """Serialize as a signed string.

        :param branch: The source branch, to get the signing strategy
        :return: a string
        """
    my_gpg = gpg.GPGStrategy(branch.get_config_stack())
    return my_gpg.sign(b''.join(self.to_lines()), gpg.MODE_CLEAR)