import argparse
import bisect
import os
import sys
import statistics
from typing import Dict, List, Set
from . import fitz
def page_blocksort(page, textout, GRID, fontsize, noformfeed, skip_empty, flags):
    eop = b'\n' if noformfeed else bytes([12])
    blocks = page.get_text('blocks', flags=flags)
    if blocks == []:
        if not skip_empty:
            textout.write(eop)
        return
    blocks.sort(key=lambda b: (b[3], b[0]))
    for b in blocks:
        textout.write(b[4].encode('utf8', errors='surrogatepass'))
    textout.write(eop)
    return