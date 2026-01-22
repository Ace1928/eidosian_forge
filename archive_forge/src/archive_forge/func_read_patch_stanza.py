import re
from .. import osutils
from ..iterablefile import IterableFile
def read_patch_stanza(line_iter):
    """Convert an iterable of RIO-Patch format lines into a Stanza.

    RIO-Patch is a RIO variant designed to be e-mailed as part of a patch.
    It resists common forms of damage such as newline conversion or the removal
    of trailing whitespace, yet is also reasonably easy to read.

    :return: a Stanza
    """
    return read_stanza(_patch_stanza_iter(line_iter))