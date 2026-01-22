import abc
import logging
from cliff import command
from cliff import lister
from cliff import show
from osc_lib import exceptions
from osc_lib.i18n import _
def deprecated_option_warning(self, old_option, new_option):
    """Emit a warning for use of a deprecated option"""
    self.log.warning(_('The %(old)s option is deprecated, please use %(new)s instead.') % {'old': old_option, 'new': new_option})