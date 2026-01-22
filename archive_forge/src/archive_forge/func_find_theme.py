import sys
import os
import re
import docutils
from docutils import frontend, nodes, utils
from docutils.writers import html4css1
from docutils.parsers.rst import directives
def find_theme(name):
    path = os.path.join(themes_dir_path, name)
    if not os.path.isdir(path):
        raise docutils.ApplicationError('Theme directory not found: %r (path: %r)' % (name, path))
    return path