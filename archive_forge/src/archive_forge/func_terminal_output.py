import os
import subprocess as sp
import shlex
import simplejson as json
from traits.trait_errors import TraitError
from ... import config, logging, LooseVersion
from ...utils.provenance import write_provenance
from ...utils.misc import str2bool
from ...utils.filemanip import (
from ...utils.subprocess import run_command
from ...external.due import due
from .traits_extension import traits, isdefined, Undefined
from .specs import (
from .support import (
@terminal_output.setter
def terminal_output(self, value):
    if value not in VALID_TERMINAL_OUTPUT:
        raise RuntimeError('Setting invalid value "%s" for terminal_output. Valid values are %s.' % (value, ', '.join(['"%s"' % v for v in VALID_TERMINAL_OUTPUT])))
    self._terminal_output = value