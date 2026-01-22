from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from six.moves import range
from prompt_toolkit.token import Token
from prompt_toolkit.filters import to_cli_filter
from .utils import split_lines
import re
import six
def get_syntax_sync():
    """ The Syntax synchronisation objcet that we currently use. """
    if self.sync_from_start(cli):
        return SyncFromStart()
    else:
        return self.syntax_sync