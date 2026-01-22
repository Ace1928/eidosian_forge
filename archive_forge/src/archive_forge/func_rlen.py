from suds import *
from suds.umx import *
from suds.sax import Namespace
def rlen(self):
    """
        Get the number of I{real} attributes which exclude xs and xml attributes.
        @return: A count of I{real} attributes.
        @rtype: L{int}
        """
    n = 0
    for a in self.real():
        n += 1
    return n