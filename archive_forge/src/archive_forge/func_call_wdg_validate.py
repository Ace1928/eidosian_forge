from io import StringIO
import subprocess
from paste.response import header_value
import re
import cgi
def call_wdg_validate(self, wdg_path, ops, page):
    if subprocess is None:
        raise ValueError('This middleware requires the subprocess module from Python 2.4')
    proc = subprocess.Popen([wdg_path] + ops, shell=False, close_fds=True, stdout=subprocess.PIPE, stdin=subprocess.PIPE, stderr=subprocess.STDOUT)
    stdout = proc.communicate(page)[0]
    proc.wait()
    return stdout