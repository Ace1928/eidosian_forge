import os
from testtools import content
from .. import plugins as _mod_plugins
from .. import trace
from ..bzr.smart import medium
from ..controldir import ControlDir
from ..transport import remote
from . import TestCaseWithTransport
def test_simple_local_bzr(self):
    self.make_branch_and_tree('.', format='bzr')
    forbidden_modules = ['breezy.annotate', 'breezy.atomicfile', 'breezy.bugtracker', 'breezy.bundle.commands', 'breezy.cmd_version_info', 'breezy.externalcommand', 'breezy.filters', 'breezy.gpg', 'breezy.info', 'breezy.bzr.knit', 'breezy.merge_directive', 'breezy.msgeditor', 'breezy.rules', 'breezy.sign_my_commits', 'breezy.transform', 'breezy.version_info_formats.format_rio', 'breezy.bzr.hashcache', 'breezy.bzr.remote', 'breezy.bzr.smart', 'breezy.bzr.smart.client', 'breezy.bzr.smart.medium', 'breezy.bzr.smart.server', 'breezy.bzr.xml_serializer', 'breezy.bzr.xml8', 'getpass', 'kerberos', 'merge3', 'shutil', 'ssl', 'socket', 'smtplib', 'tarfile', 'termios', 'tty', 'ctypes'] + old_format_modules
    self.run_command_check_imports(['st'], forbidden_modules)