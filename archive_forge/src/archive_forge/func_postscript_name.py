from collections import namedtuple
import logging
import re
from ._mathtext_data import uni2type1
@property
def postscript_name(self):
    return self.get_fontname()