import re
from codecs import BOM_UTF8
from typing import Tuple
from parso.python.tokenize import group
def create_spacing_part(self):
    column = self.start_pos[1] - len(self.spacing)
    return PrefixPart(self.parent, 'spacing', self.spacing, start_pos=(self.start_pos[0], column))