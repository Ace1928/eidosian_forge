import builtins
import configparser
import operator
import sys
from cherrypy._cpcompat import text_or_bytes
def dict_from_file(self, file):
    if hasattr(file, 'read'):
        self.read_file(file)
    else:
        self.read(file)
    return self.as_dict()