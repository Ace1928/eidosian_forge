from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_setTypes(self):
    """
        If the I{libc} object passed to L{initializeModule} has all of the
        necessary attributes, it sets the C{argtypes} and C{restype} attributes
        of the three ctypes methods used from libc.
        """

    class libc:

        def inotify_init(self):
            pass
        inotify_init = staticmethod(inotify_init)

        def inotify_rm_watch(self):
            pass
        inotify_rm_watch = staticmethod(inotify_rm_watch)

        def inotify_add_watch(self):
            pass
        inotify_add_watch = staticmethod(inotify_add_watch)
    c = libc()
    initializeModule(c)
    self.assertEqual(c.inotify_init.argtypes, [])
    self.assertEqual(c.inotify_init.restype, c_int)
    self.assertEqual(c.inotify_rm_watch.argtypes, [c_int, c_int])
    self.assertEqual(c.inotify_rm_watch.restype, c_int)
    self.assertEqual(c.inotify_add_watch.argtypes, [c_int, c_char_p, c_uint32])
    self.assertEqual(c.inotify_add_watch.restype, c_int)