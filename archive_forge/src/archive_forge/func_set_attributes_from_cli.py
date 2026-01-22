from __future__ import (absolute_import, division, print_function)
from ansible import constants as C
from ansible import context
from ansible.playbook.attribute import FieldAttribute
from ansible.playbook.base import Base
from ansible.utils.display import Display
def set_attributes_from_cli(self):
    """
        Configures this connection information instance with data from
        options specified by the user on the command line. These have a
        lower precedence than those set on the play or host.
        """
    if context.CLIARGS.get('timeout', False):
        self.timeout = int(context.CLIARGS['timeout'])
    self.private_key_file = context.CLIARGS.get('private_key_file')
    self._internal_verbosity = context.CLIARGS.get('verbosity')
    self.start_at_task = context.CLIARGS.get('start_at_task', None)