from __future__ import annotations
from . import Extension
from ..blockprocessors import BlockProcessor
from ..inlinepatterns import InlineProcessor
from ..treeprocessors import Treeprocessor
from ..postprocessors import Postprocessor
from .. import util
from collections import OrderedDict
import re
import copy
import xml.etree.ElementTree as etree
def findFootnotesPlaceholder(self, root: etree.Element) -> tuple[etree.Element, etree.Element, bool] | None:
    """ Return ElementTree Element that contains Footnote placeholder. """

    def finder(element):
        for child in element:
            if child.text:
                if child.text.find(self.getConfig('PLACE_MARKER')) > -1:
                    return (child, element, True)
            if child.tail:
                if child.tail.find(self.getConfig('PLACE_MARKER')) > -1:
                    return (child, element, False)
            child_res = finder(child)
            if child_res is not None:
                return child_res
        return None
    res = finder(root)
    return res