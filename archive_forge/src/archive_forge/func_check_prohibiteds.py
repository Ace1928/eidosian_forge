import stringprep
from encodings import idna
from itertools import chain
from unicodedata import ucd_3_2_0 as unicodedata
from zope.interface import Interface, implementer
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
def check_prohibiteds(self, string):
    for c in string:
        if c in self.prohibiteds:
            raise UnicodeError('Invalid character %s' % repr(c))