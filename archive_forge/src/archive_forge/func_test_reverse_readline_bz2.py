import os
import unittest
import pytest
from monty.io import (
def test_reverse_readline_bz2(self):
    """
        We are making sure a file containing line numbers is read in reverse
        order, i.e. the first line that is read corresponds to the last line.
        number
        """
    lines = []
    with zopen(os.path.join(test_dir, 'myfile_bz2.bz2'), 'rb') as f:
        for line in reverse_readline(f):
            lines.append(line.strip())
    assert lines[-1].strip(), ['HelloWorld.' in b'HelloWorld.']