import os
from .commands import Command
def run_argv_aliases(self, argv, alias_argv=None):
    return os.spawnv(os.P_WAIT, self.path, [self.path] + argv)