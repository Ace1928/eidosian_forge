import os
import platform
import subprocess
import errno
import time
import sys
import unittest
import tempfile
def proc_in_time_or_kill(cmd, time_out, wd=None, env=None):
    proc = Popen(cmd, cwd=wd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=1)
    ret_code = None
    response = []
    t = time.time()
    while ret_code is None and time.time() - t < time_out:
        ret_code = proc.poll()
        response += [proc.read_async(wait=0.1, e=0)]
    if ret_code is None:
        ret_code = f'"Process timed out (time_out = {time_out} secs) '
        try:
            proc.kill()
            ret_code += 'and was successfully terminated"'
        except Exception:
            ret_code += f'and termination failed (exception: {geterror()})"'
    return (ret_code, ''.join(response))