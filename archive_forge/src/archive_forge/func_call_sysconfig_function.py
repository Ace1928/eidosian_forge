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
def call_sysconfig_function(self, name, *args, **kw):
    """
        Call the specified function in the first sysconfig module it
        is defined in.  ``NameError`` if no function is found.
        """
    val = self.get_sysconfig_option(name)
    if val is None:
        raise NameError('Method %s not found in any sysconfig module' % name)
    return val(*args, **kw)