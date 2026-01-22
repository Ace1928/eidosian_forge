import re
import sys
import typing
from .util import (
from .unicode import pyparsing_unicode as ppu
@parserElement.setter
def parserElement(self, elem):
    self.parser_element = elem