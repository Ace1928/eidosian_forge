import re
from lxml import etree
from .jsonutil import JsonTable
def element_keys(self, name):
    """ Returns the attribute keys of this specific element.
        """
    keys = set()
    for element in self.__call__('//%s' % name):
        for element_key in element.keys():
            keys.add(element_key)
    return list(keys)