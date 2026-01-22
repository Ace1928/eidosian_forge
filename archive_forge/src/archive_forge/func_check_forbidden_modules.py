import os
from testtools import content
from .. import plugins as _mod_plugins
from .. import trace
from ..bzr.smart import medium
from ..controldir import ControlDir
from ..transport import remote
from . import TestCaseWithTransport
def check_forbidden_modules(self, err, forbidden_imports):
    """Check for forbidden modules in stderr.

        :param err: Standard error
        :param forbidden_imports: List of forbidden modules
        """
    err = err.decode('utf-8')
    self.addDetail('subprocess_stderr', content.text_content(err))
    bad_modules = []
    for module_name in forbidden_imports:
        if err.find("\nimport '%s' " % module_name) != -1:
            bad_modules.append(module_name)
    if bad_modules:
        self.fail('command loaded forbidden modules %r' % (bad_modules,))