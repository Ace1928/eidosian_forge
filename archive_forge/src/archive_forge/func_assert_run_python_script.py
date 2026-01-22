import sys
import os
import os.path as op
import tempfile
from subprocess import Popen, check_output, PIPE, STDOUT, CalledProcessError
from srsly.cloudpickle.compat import pickle
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor
import psutil
from srsly.cloudpickle import dumps
from subprocess import TimeoutExpired
def assert_run_python_script(source_code, timeout=TIMEOUT):
    """Utility to help check pickleability of objects defined in __main__

    The script provided in the source code should return 0 and not print
    anything on stderr or stdout.
    """
    fd, source_file = tempfile.mkstemp(suffix='_src_test_cloudpickle.py')
    os.close(fd)
    try:
        with open(source_file, 'wb') as f:
            f.write(source_code.encode('utf-8'))
        cmd = [sys.executable, '-W ignore', source_file]
        cwd, env = _make_cwd_env()
        kwargs = {'cwd': cwd, 'stderr': STDOUT, 'env': env}
        coverage_rc = os.environ.get('COVERAGE_PROCESS_START')
        if coverage_rc:
            kwargs['env']['COVERAGE_PROCESS_START'] = coverage_rc
        kwargs['timeout'] = timeout
        try:
            try:
                out = check_output(cmd, **kwargs)
            except CalledProcessError as e:
                raise RuntimeError('script errored with output:\n%s' % e.output.decode('utf-8')) from e
            if out != b'':
                raise AssertionError(out.decode('utf-8'))
        except TimeoutExpired as e:
            raise RuntimeError('script timeout, output so far:\n%s' % e.output.decode('utf-8')) from e
    finally:
        os.unlink(source_file)