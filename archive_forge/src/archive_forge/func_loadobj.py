from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def loadobj(object_type, uri, name=None, relative_to=None, global_conf=None):
    context = loadcontext(object_type, uri, name=name, relative_to=relative_to, global_conf=global_conf)
    return context.create()