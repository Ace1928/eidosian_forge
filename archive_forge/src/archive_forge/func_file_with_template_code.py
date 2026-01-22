import contextlib
import pathlib
from pathlib import Path
import re
import time
from typing import Union
from unittest import mock
def file_with_template_code(filespec):
    with open(filespec, 'w') as f:
        f.write('\ni am an artificial template just for you\n')
    return filespec