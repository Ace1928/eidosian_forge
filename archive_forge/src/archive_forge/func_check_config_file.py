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
def check_config_file(self):
    if self.installer.expect_config_directory is None:
        return
    fn = self.config_file
    if self.installer.expect_config_directory:
        if os.path.splitext(fn)[1]:
            raise BadCommand('The CONFIG_FILE argument %r looks like a filename, and a directory name is expected' % fn)
    elif fn.endswith('/') or not os.path.splitext(fn):
        raise BadCommand('The CONFIG_FILE argument %r looks like a directory name and a filename is expected' % fn)