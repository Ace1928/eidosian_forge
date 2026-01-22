import builtins
import configparser
import operator
import sys
from cherrypy._cpcompat import text_or_bytes
def build_Add(self, o):
    return operator.add