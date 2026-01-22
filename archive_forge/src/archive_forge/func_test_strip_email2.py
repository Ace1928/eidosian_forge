import os
import math
import random
from pathlib import Path
import pytest
from IPython.utils import text
def test_strip_email2():
    src = '> > > list()'
    cln = 'list()'
    assert text.strip_email_quotes(src) == cln