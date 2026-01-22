from __future__ import absolute_import
import sys
import subprocess
import time
from twisted.trial.unittest import TestCase
from crochet._shutdown import (
from ..tests import crochet_directory
import threading, sys
from crochet._shutdown import register, _watchdog
def test_shutdown(self):
    """
        A function registered with _shutdown.register() is called when the
        main thread exits.
        """
    program = 'import threading, sys\n\nfrom crochet._shutdown import register, _watchdog\n_watchdog.start()\n\nend = False\n\ndef thread():\n    while not end:\n        pass\n    sys.stdout.write("byebye")\n    sys.stdout.flush()\n\ndef stop(x, y):\n    # Move this into separate test at some point.\n    assert x == 1\n    assert y == 2\n    global end\n    end = True\n\nthreading.Thread(target=thread).start()\nregister(stop, 1, y=2)\n\nsys.exit()\n'
    process = subprocess.Popen([sys.executable, '-c', program], cwd=crochet_directory, stdout=subprocess.PIPE)
    result = process.stdout.read()
    self.assertEqual(process.wait(), 0)
    self.assertEqual(result, b'byebye')