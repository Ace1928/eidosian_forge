import html.entities
import re
from .sgml import *
@staticmethod
def convert_charref(name):
    """
        :type name: str
        :rtype: str
        """
    return '&#%s;' % name