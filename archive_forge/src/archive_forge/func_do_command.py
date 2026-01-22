from typing import List
from breezy import branch, urlutils
from breezy.tests import script
def do_command(self, *args):
    out, err = self.run_bzr(self.command + list(args), working_dir=self.working_dir)