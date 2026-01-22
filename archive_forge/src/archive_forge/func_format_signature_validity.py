import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
def format_signature_validity(rev_id, branch):
    """get the signature validity

    :param rev_id: revision id to validate
    :param branch: branch of revision
    :return: human readable string to print to log
    """
    from breezy import gpg
    gpg_strategy = gpg.GPGStrategy(branch.get_config_stack())
    result = branch.repository.verify_revision_signature(rev_id, gpg_strategy)
    if result[0] == gpg.SIGNATURE_VALID:
        return f'valid signature from {result[1]}'
    if result[0] == gpg.SIGNATURE_KEY_MISSING:
        return f'unknown key {result[1]}'
    if result[0] == gpg.SIGNATURE_NOT_VALID:
        return 'invalid signature!'
    if result[0] == gpg.SIGNATURE_NOT_SIGNED:
        return 'no signature'