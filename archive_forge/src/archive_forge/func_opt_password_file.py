import warnings
from twisted.application import internet
from twisted.cred import checkers, portal, strcred
from twisted.protocols import ftp
from twisted.python import deprecate, usage, versions
def opt_password_file(self, filename):
    """
        Specify a file containing username:password login info for
        authenticated connections. (DEPRECATED; see --help-auth instead)
        """
    self['password-file'] = filename
    msg = deprecate.getDeprecationWarningString(self.opt_password_file, versions.Version('Twisted', 11, 1, 0))
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)
    self.addChecker(checkers.FilePasswordDB(filename, cache=True))