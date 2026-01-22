from __future__ import unicode_literals
import logging
import re
from cmakelang.parse import util as parse_util
from cmakelang.parse.funs import standard_funs
from cmakelang import markup
from cmakelang.config_util import (
def set_line_ending(self, detected):
    self.endl = {'windows': '\r\n', 'unix': '\n'}[detected]