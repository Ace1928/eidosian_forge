import os
import sys
import hashlib
import filecmp
from collections import namedtuple
from shutil import which
from jedi.cache import memoize_method, time_cache
from jedi.inference.compiled.subprocess import CompiledSubprocess, \
import parso
@memoize_method
def get_grammar(self):
    version_string = '%s.%s' % (self.version_info.major, self.version_info.minor)
    return parso.load_grammar(version=version_string)