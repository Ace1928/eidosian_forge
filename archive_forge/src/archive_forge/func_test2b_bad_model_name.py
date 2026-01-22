from itertools import zip_longest
import json
import re
import os
import sys
from os.path import abspath, dirname, join
from filecmp import cmp
import subprocess
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml_available
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tee import capture_output
from pyomo.common.tempfiles import TempfileManager
import pyomo.core
import pyomo.scripting.pyomo_main as main
from pyomo.opt import check_available_solvers
from io import StringIO
def test2b_bad_model_name(self):
    self.pyomo(join(currdir, 'test2b.json'), root=join(currdir, 'test2'))
    self.compare_files(join(currdir, 'test2.out'), join(currdir, 'test2.txt'))