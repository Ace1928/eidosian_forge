import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
@_Cache.me
def cc_test_flags(self, flags):
    """
        Returns True if the compiler supports 'flags'.
        """
    assert isinstance(flags, list)
    self.dist_log('testing flags', flags)
    test_path = os.path.join(self.conf_check_path, 'test_flags.c')
    test = self.dist_test(test_path, flags)
    if not test:
        self.dist_log('testing failed', stderr=True)
    return test