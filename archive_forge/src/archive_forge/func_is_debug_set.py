import inspect
import io
import logging
import re
import sys
import textwrap
from pyomo.version.info import releaselevel
from pyomo.common.deprecation import deprecated
from pyomo.common.fileutils import PYOMO_ROOT_DIR
from pyomo.common.formatting import wrap_reStructuredText
def is_debug_set(logger):
    if not logger.isEnabledFor(_DEBUG):
        return False
    return logger.getEffectiveLevel() > _NOTSET