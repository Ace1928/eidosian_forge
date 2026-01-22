from a disk file or from an open file, and similar for its output.
import re
import os
import tempfile
import warnings
from shlex import quote
t.open_r(file) and t.open_w(file) implement
        t.open(file, 'r') and t.open(file, 'w') respectively.