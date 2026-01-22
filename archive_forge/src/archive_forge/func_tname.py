import os
from typing import Dict, Type
def tname(self, table):
    """Return the name of the table."""
    if table != 'biosequence':
        return table
    else:
        return 'bioentry'