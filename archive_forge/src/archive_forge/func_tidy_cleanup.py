import six
import subprocess
from ebooklib.plugins.base import BasePlugin
from ebooklib.utils import parse_html_string
def tidy_cleanup(content, **extra):
    cmd = []
    for k, v in six.iteritems(extra):
        if v:
            cmd.append('--%s' % k)
            cmd.append(v)
        else:
            cmd.append('-%s' % k)
    try:
        p = subprocess.Popen(['tidy'] + cmd, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    except OSError:
        return (3, None)
    p.stdin.write(content)
    cont, p_err = p.communicate()
    return (p.returncode, cont)