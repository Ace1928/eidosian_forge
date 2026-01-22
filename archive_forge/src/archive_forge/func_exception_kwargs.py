import re
from mako import ast
from mako import exceptions
from mako import filters
from mako import util
@property
def exception_kwargs(self):
    return {'source': self.source, 'lineno': self.lineno, 'pos': self.pos, 'filename': self.filename}