import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def joinligature(lig: str) -> str:
    """Return ligature character for a given pair / triple of characters.

        Args:
            lig: (str) 2/3 characters, e.g. "ff"
        Returns:
            Ligature, e.g. "ff" -> chr(0xFB00)
        """
    if lig == 'ff':
        return chr(64256)
    elif lig == 'fi':
        return chr(64257)
    elif lig == 'fl':
        return chr(64258)
    elif lig == 'ffi':
        return chr(64259)
    elif lig == 'ffl':
        return chr(64260)
    elif lig == 'ft':
        return chr(64261)
    elif lig == 'st':
        return chr(64262)
    return lig