from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def loadfilter(uri, name=None, **kw):
    return loadobj(FILTER, uri, name=name, **kw)