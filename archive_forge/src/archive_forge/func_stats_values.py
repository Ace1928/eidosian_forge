import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def stats_values(self):
    """Calculate the number of inserts and removes."""
    removes = 0
    inserts = 0
    for hunk in self.hunks:
        for line in hunk.lines:
            if isinstance(line, InsertLine):
                inserts += 1
            elif isinstance(line, RemoveLine):
                removes += 1
    return (inserts, removes, len(self.hunks))