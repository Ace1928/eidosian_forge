import sys
import os
import shutil
import io
import re
import textwrap
from os.path import relpath
from errno import EEXIST
import traceback
def format_template(template, **kw):
    return jinja.from_string(template, **kw)