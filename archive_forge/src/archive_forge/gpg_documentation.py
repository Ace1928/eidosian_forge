import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
Set the acceptable keys for verifying with this GPGStrategy.

        :param command_line_input: comma separated list of patterns from
                                command line
        :return: nothing
        