import os
import sys
from io import BytesIO
from typing import Callable, Dict, Iterable, Tuple, cast
import configobj
import breezy
from .lazy_import import lazy_import
import errno
import fnmatch
import re
from breezy import (
from breezy.i18n import gettext
from . import (bedding, commands, errors, hooks, lazy_regex, registry, trace,
from .option import Option as CommandOption
def get_merge_tools(self):
    tools = {}
    for oname, value, section, conf_id, parser in self._get_options():
        if oname.startswith('bzr.mergetool.'):
            tool_name = oname[len('bzr.mergetool.'):]
            tools[tool_name] = self.get_user_option(oname, False)
    trace.mutter('loaded merge tools: %r' % tools)
    return tools