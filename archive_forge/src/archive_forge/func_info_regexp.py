import os
import re
import shutil
from ... import logging
from ...utils.filemanip import fname_presuffix, split_filename
from ..base import (
from .base import (
def info_regexp(self, info, field, delim='\n'):
    m = re.search('%s\\s*:\\s+(.+?)%s' % (field, delim), info)
    if m:
        return m.group(1)
    else:
        return None