import os.path
import re
import unittest
import idna
def parse_idna_test_table(inputstream):
    """Parse IdnaTestV2.txt and return a list of tuples."""
    for lineno, line in enumerate(inputstream):
        line = line.decode('utf-8').strip()
        if '#' in line:
            line = line.split('#', 1)[0]
        if not line:
            continue
        yield (lineno + 1, tuple((field.strip() for field in line.split(';'))))