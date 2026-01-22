import os
import sys
import py
import tempfile
def startall(self):
    if self.out:
        sys.stdout = self.out
    if self.err:
        sys.stderr = self.err
    if self.in_:
        sys.stdin = self.in_ = DontReadFromInput()