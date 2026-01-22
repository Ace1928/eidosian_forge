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
def simple_config(self, vars):
    """
        Return a very simple configuration file for this application.
        """
    if self.ep_name != 'main':
        ep_name = '#' + self.ep_name
    else:
        ep_name = ''
    return '[app:main]\nuse = egg:%s%s\n' % (self.dist.project_name, ep_name)