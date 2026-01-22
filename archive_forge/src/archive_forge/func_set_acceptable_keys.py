import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
def set_acceptable_keys(self, command_line_input):
    """Set the acceptable keys for verifying with this GPGStrategy.

        :param command_line_input: comma separated list of patterns from
                                command line
        :return: nothing
        """
    patterns = None
    acceptable_keys_config = self._config_stack.get('acceptable_keys')
    if acceptable_keys_config is not None:
        patterns = acceptable_keys_config
    if command_line_input is not None:
        patterns = command_line_input.split(',')
    if patterns:
        self.acceptable_keys = []
        for pattern in patterns:
            result = self.context.keylist(pattern)
            found_key = False
            for key in result:
                found_key = True
                self.acceptable_keys.append(key.subkeys[0].fpr)
                trace.mutter('Added acceptable key: ' + key.subkeys[0].fpr)
            if not found_key:
                trace.note(gettext('No GnuPG key results for pattern: {0}').format(pattern))