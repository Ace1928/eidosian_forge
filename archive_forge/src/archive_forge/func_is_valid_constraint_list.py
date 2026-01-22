import logging
import re
from .compat import string_types
from .util import parse_requirement
def is_valid_constraint_list(self, s):
    """
        Used for processing some metadata fields
        """
    if s.endswith(','):
        s = s[:-1]
    return self.is_valid_matcher('dummy_name (%s)' % s)