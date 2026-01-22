import sys
import os
import shutil
import io
import re
import textwrap
from os.path import relpath
from errno import EEXIST
import traceback
def remove_coding(text):
    """
    Remove the coding comment, which exec doesn't like.
    """
    sub_re = re.compile('^#\\s*-\\*-\\s*coding:\\s*.*-\\*-$', flags=re.MULTILINE)
    return sub_re.sub('', text)