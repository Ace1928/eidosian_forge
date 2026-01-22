from suds import *
from suds.umx import *
from suds.umx.attrlist import AttrList
from suds.sax.text import Text
from suds.sudsobject import Factory, merge
def single_occurrence(self, content):
    """
        Get whether the content has at most a single occurrence (not a list).
        @param content: The current content being unmarshalled.
        @type content: L{Content}
        @return: True if content has at most a single occurrence, else False.
        @rtype: boolean
        '"""
    return not self.multi_occurrence(content)