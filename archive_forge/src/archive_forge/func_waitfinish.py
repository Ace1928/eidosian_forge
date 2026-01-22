import py
import os
import sys
import marshal
def waitfinish(self, waiter=os.waitpid):
    pid, systemstatus = waiter(self.pid, 0)
    if systemstatus:
        if os.WIFSIGNALED(systemstatus):
            exitstatus = os.WTERMSIG(systemstatus) + 128
        else:
            exitstatus = os.WEXITSTATUS(systemstatus)
    else:
        exitstatus = 0
    signal = systemstatus & 127
    if not exitstatus and (not signal):
        retval = self.RETVAL.open('rb')
        try:
            retval_data = retval.read()
        finally:
            retval.close()
        retval = marshal.loads(retval_data)
    else:
        retval = None
    stdout = self.STDOUT.read()
    stderr = self.STDERR.read()
    self._removetemp()
    return Result(exitstatus, signal, retval, stdout, stderr)