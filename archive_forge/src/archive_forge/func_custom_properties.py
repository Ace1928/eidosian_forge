import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
def custom_properties(self, revision):
    """Format the custom properties returned by each registered handler.

        If a registered handler raises an error it is propagated.

        :return: a list of formatted lines (excluding trailing newlines)
        """
    lines = self._foreign_info_properties(revision)
    for key, handler in properties_handler_registry.iteritems():
        try:
            lines.extend(self._format_properties(handler(revision)))
        except Exception:
            trace.log_exception_quietly()
            trace.print_exception(sys.exc_info(), self.to_file)
    return lines