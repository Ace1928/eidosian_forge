import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
def get_type_by_name(path):
    """Returns type of file by its name, or None if not known"""
    update_cache()
    return globs.first_match(path)