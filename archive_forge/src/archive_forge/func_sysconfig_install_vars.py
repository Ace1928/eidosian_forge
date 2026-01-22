import types
import os
import string
import uuid
from paste.deploy import appconfig
from paste.script import copydir
from paste.script.command import Command, BadCommand, run as run_command
from paste.script.util import secret
from paste.util import import_string
import paste.script.templates
import pkg_resources
def sysconfig_install_vars(self, installer):
    """
        Return the folded results of calling the
        ``install_variables()`` functions.
        """
    result = {}
    all_vars = self.call_sysconfig_functions('install_variables', installer)
    all_vars.reverse()
    for vardict in all_vars:
        result.update(vardict)
    return result