from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_missingAdd(self):
    """
        If the I{libc} object passed to L{initializeModule} has no
        C{inotify_add_watch} attribute, L{ImportError} is raised.
        """

    class libc:

        def inotify_init(self):
            pass

        def inotify_rm_watch(self):
            pass
    self.assertRaises(ImportError, initializeModule, libc())