import subprocess
import sys
from twisted.trial.unittest import TestCase
from twisted.python.runtime import platform
from ..tests import crochet_directory
from crochet import setup, run_in_reactor
import sys
import os
from twisted.internet.protocol import ProcessProtocol
from twisted.internet.defer import Deferred
from twisted.internet import reactor
def test_processExit(self):
    """
        A Crochet-managed reactor notice when a process it started exits.

        On POSIX platforms this requies waitpid() to be called, which in
        default Twisted implementation relies on a SIGCHLD handler which is not
        installed by Crochet at the moment.
        """
    program = 'from crochet import setup, run_in_reactor\nsetup()\n\nimport sys\nimport os\nfrom twisted.internet.protocol import ProcessProtocol\nfrom twisted.internet.defer import Deferred\nfrom twisted.internet import reactor\n\nclass Waiter(ProcessProtocol):\n    def __init__(self):\n        self.result = Deferred()\n\n    def processExited(self, reason):\n        self.result.callback(None)\n\n\n@run_in_reactor\ndef run():\n    waiter = Waiter()\n    # Closing FDs before exit forces us to rely on SIGCHLD to notice process\n    # exit:\n    reactor.spawnProcess(waiter, sys.executable,\n                         [sys.executable, \'-c\',\n                          \'import os; os.close(0); os.close(1); os.close(2)\'],\n                         env=os.environ)\n    return waiter.result\n\nrun().wait(10)\n# If we don\'t notice process exit, TimeoutError will be thrown and we won\'t\n# reach the next line:\nsys.stdout.write("abc")\n'
    process = subprocess.Popen([sys.executable, '-c', program], cwd=crochet_directory, stdout=subprocess.PIPE)
    result = process.stdout.read()
    self.assertEqual(result, b'abc')