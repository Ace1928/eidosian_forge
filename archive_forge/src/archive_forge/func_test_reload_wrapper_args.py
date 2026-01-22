import os
import shutil
import subprocess
from subprocess import Popen
import sys
from tempfile import mkdtemp
import textwrap
import time
import unittest
import sys
import sys
import testapp
import os
import sys
import os
import sys
def test_reload_wrapper_args(self):
    main = 'import os\nimport sys\n\nprint(os.path.basename(sys.argv[0]))\nprint(f\'argv={sys.argv[1:]}\')\nexec(open("run_twice_magic.py").read())\n'
    self.write_files({'main.py': main})
    out = self.run_subprocess([sys.executable, '-m', 'tornado.autoreload', 'main.py', 'arg1', '--arg2', '-m', 'arg3'])
    self.assertEqual(out, "main.py\nargv=['arg1', '--arg2', '-m', 'arg3']\n" * 2)