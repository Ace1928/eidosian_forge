from gitdb.util import bin_to_hex
from gitdb.fun import (
@property
def pack_offset(self):
    return self[0]