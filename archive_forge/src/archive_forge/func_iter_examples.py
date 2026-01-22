import keyword
import os
import re
import subprocess
import sys
from taskflow import test
def iter_examples():
    examples_dir = root_path('taskflow', 'examples')
    for filename in os.listdir(examples_dir):
        path = os.path.join(examples_dir, filename)
        if not os.path.isfile(path):
            continue
        name, ext = os.path.splitext(filename)
        if ext != '.py':
            continue
        if not name.endswith('utils'):
            safe_name = safe_filename(name)
            if safe_name:
                yield (name, safe_name)