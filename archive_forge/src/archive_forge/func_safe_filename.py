import keyword
import os
import re
import subprocess
import sys
from taskflow import test
def safe_filename(filename):
    name = re.sub('[^a-zA-Z0-9_]+', '_', filename)
    if not name or re.match('^[_]+$', name) or keyword.iskeyword(name):
        return False
    return name