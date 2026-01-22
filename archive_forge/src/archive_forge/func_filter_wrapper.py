from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def filter_wrapper(wsgi_app):
    return fix_call(context.object, wsgi_app, context.global_conf, **context.local_conf)