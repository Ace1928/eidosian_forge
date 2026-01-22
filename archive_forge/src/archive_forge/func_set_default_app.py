import subprocess
import doctest
import os
import sys
import shutil
import re
import cgi
import rfc822
from io import StringIO
from paste.util import PySourceColor
def set_default_app(app, url):
    global default_app
    global default_url
    default_app = app
    default_url = url