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
def test_reload_wrapper_until_success(self):
    main = 'import os\nimport sys\n\nif "TESTAPP_STARTED" in os.environ:\n    print("exiting cleanly")\n    sys.exit(0)\nelse:\n    print("reloading")\n    exec(open("run_twice_magic.py").read())\n'
    self.write_files({'main.py': main})
    out = self.run_subprocess([sys.executable, '-m', 'tornado.autoreload', '--until-success', 'main.py'])
    self.assertEqual(out, 'reloading\nexiting cleanly\n')