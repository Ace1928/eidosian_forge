import faulthandler
import itertools
import multiprocessing
import os
import random
import re
import subprocess
import sys
import textwrap
import threading
import unittest
import numpy as np
from numba import jit, vectorize, guvectorize, set_num_threads
from numba.tests.support import (temp_directory, override_config, TestCase, tag,
import queue as t_queue
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
from numba.core import config
@linux_only
def test_lifetime_of_task_scheduler_handle(self):
    self.skip_if_no_external_compiler()
    BROKEN_COMPILERS = 'SKIP: COMPILATION FAILED'
    runme = 'if 1:\n            import ctypes\n            import sys\n            import multiprocessing as mp\n            from tempfile import TemporaryDirectory, NamedTemporaryFile\n            from numba.pycc.platform import Toolchain, external_compiler_works\n            from numba import njit, prange, threading_layer\n            import faulthandler\n            faulthandler.enable()\n            if not external_compiler_works():\n                raise AssertionError(\'External compilers are not found.\')\n            with TemporaryDirectory() as tmpdir:\n                with NamedTemporaryFile(dir=tmpdir) as tmpfile:\n                    try:\n                        src = """\n                        #define TBB_PREVIEW_WAITING_FOR_WORKERS 1\n                        #include <tbb/tbb.h>\n                        static tbb::task_scheduler_handle tsh;\n                        extern "C"\n                        {\n                        void launch(void)\n                        {\n                            tsh = tbb::task_scheduler_handle::get();\n                        }\n                        }\n                        """\n                        cxxfile = f"{tmpfile.name}.cxx"\n                        with open(cxxfile, \'wt\') as f:\n                            f.write(src)\n                        tc = Toolchain()\n                        object_files = tc.compile_objects([cxxfile,],\n                                                           output_dir=tmpdir)\n                        dso_name = f"{tmpfile.name}.so"\n                        tc.link_shared(dso_name, object_files,\n                                       libraries=[\'tbb\',],\n                                       export_symbols=[\'launch\'])\n                        # Load into the process, it doesn\'t matter whether the\n                        # DSO exists on disk once it\'s loaded in.\n                        DLL = ctypes.CDLL(dso_name)\n                    except Exception as e:\n                        # Something is broken in compilation, could be one of\n                        # many things including, but not limited to: missing tbb\n                        # headers, incorrect permissions, compilers that don\'t\n                        # work for the above\n                        print(e)\n                        print(\'BROKEN_COMPILERS\')\n                        sys.exit(0)\n\n                    # Do the test, launch this library and also execute a\n                    # function with the TBB threading layer.\n\n                    DLL.launch()\n\n                    @njit(parallel=True)\n                    def foo(n):\n                        acc = 0\n                        for i in prange(n):\n                            acc += i\n                        return acc\n\n                    foo(1)\n\n            # Check the threading layer used was TBB\n            assert threading_layer() == \'tbb\'\n\n            # Use mp context for a controlled version of fork, this triggers the\n            # reported bug.\n\n            ctx = mp.get_context(\'fork\')\n            def nowork():\n                pass\n            p = ctx.Process(target=nowork)\n            p.start()\n            p.join(10)\n            print("SUCCESS")\n            '.replace('BROKEN_COMPILERS', BROKEN_COMPILERS)
    cmdline = [sys.executable, '-c', runme]
    env = os.environ.copy()
    env['NUMBA_THREADING_LAYER'] = 'tbb'
    out, err = self.run_cmd(cmdline, env=env)
    if BROKEN_COMPILERS in out:
        self.skipTest('Compilation of DSO failed. Check output for details')
    else:
        self.assertIn('SUCCESS', out)
    if self._DEBUG:
        print('OUT:', out)
        print('ERR:', err)