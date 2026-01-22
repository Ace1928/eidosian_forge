import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def get_script(self):
    s = super().get_script()
    s = s.replace('$(brz ', '$(%s ' % ' '.join(self.get_brz_command()))
    s = s.replace('2>/dev/null', '')
    return s