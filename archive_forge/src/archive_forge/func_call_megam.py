import subprocess
from nltk.internals import find_binary
def call_megam(args):
    """
    Call the ``megam`` binary with the given arguments.
    """
    if isinstance(args, str):
        raise TypeError('args should be a list of strings')
    if _megam_bin is None:
        config_megam()
    cmd = [_megam_bin] + args
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    stdout, stderr = p.communicate()
    if p.returncode != 0:
        print()
        print(stderr)
        raise OSError('megam command failed!')
    if isinstance(stdout, str):
        return stdout
    else:
        return stdout.decode('utf-8')