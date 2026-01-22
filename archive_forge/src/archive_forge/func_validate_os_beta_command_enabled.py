import abc
import logging
from cliff import command
from cliff import lister
from cliff import show
from osc_lib import exceptions
from osc_lib.i18n import _
def validate_os_beta_command_enabled(self):
    if not self.app.options.os_beta_command:
        msg = _('Caution: This is a beta command and subject to change. Use global option --os-beta-command to enable this command.')
        raise exceptions.CommandError(msg)