import re
import warnings
from itertools import chain
from xml.etree import ElementTree
from xml.sax.saxutils import XMLGenerator, escape
from Bio import BiopythonParserWarning
from Bio.SearchIO._index import SearchIndexer
from Bio.SearchIO._model import QueryResult, Hit, HSP, HSPFragment
def startParent(self, name, attrs=None):
    """Start an XML element which has children.

        :param name: element name
        :type name: string
        :param attrs: element attributes
        :type attrs: dictionary {string: object}

        """
    if attrs is None:
        attrs = {}
    self.startElement(name, attrs, children=True)
    self._level += self._increment
    self._write('\n')
    self._parent_stack.append(name)