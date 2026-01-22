import sys
from typing import Optional, Sequence, Type
from zope.interface import Attribute, Interface
from twisted.plugin import getPlugins
from twisted.python import usage
def opt_help_auth(self):
    """
        Show all authentication methods available.
        """
    self.authOutput.write('Usage: --auth AuthType[:ArgString]\n')
    self.authOutput.write('For detailed help: --help-auth-type AuthType\n')
    self.authOutput.write('\n')
    firstLength = 0
    for factory in self._checkerFactoriesForOptHelpAuth():
        if len(factory.authType) > firstLength:
            firstLength = len(factory.authType)
    formatString = '  %%-%is\t%%s\n' % firstLength
    self.authOutput.write(formatString % ('AuthType', 'ArgString format'))
    self.authOutput.write(formatString % ('========', '================'))
    for factory in self._checkerFactoriesForOptHelpAuth():
        self.authOutput.write(formatString % (factory.authType, factory.argStringFormat))
    self.authOutput.write('\n')
    raise SystemExit(0)