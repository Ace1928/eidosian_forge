import os
import re
import sysconfig
def not_hidden(name):
    """Return True if file 'name' isn't .hidden."""
    return not name.startswith('.')