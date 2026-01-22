from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def find_entry_point(dist, group, name):
    for entry in dist.entry_points:
        if entry.name == name and entry.group == group:
            return entry