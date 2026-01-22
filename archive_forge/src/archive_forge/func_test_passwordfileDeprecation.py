from twisted.cred import credentials, error
from twisted.python import versions
from twisted.python.filepath import FilePath
from twisted.tap.ftp import Options
from twisted.trial.unittest import TestCase
def test_passwordfileDeprecation(self) -> None:
    """
        The C{--password-file} option will emit a warning stating that
        said option is deprecated.
        """
    self.callDeprecated(versions.Version('Twisted', 11, 1, 0), self.options.opt_password_file, self.filename)