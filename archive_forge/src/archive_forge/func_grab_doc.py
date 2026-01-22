from nipype.interfaces import fsl
from nipype.utils import docparse
import subprocess
from ..interfaces.base import CommandLine
from .misc import is_container
def grab_doc(cmd, trap_error=True):
    """Run cmd without args and grab documentation.

    Parameters
    ----------
    cmd : string
        Command line string
    trap_error : boolean
        Ensure that returncode is 0

    Returns
    -------
    doc : string
        The command line documentation
    """
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    stdout, stderr = proc.communicate()
    if trap_error and proc.returncode:
        msg = 'Attempting to run %s. Returned Error: %s' % (cmd, stderr)
        raise IOError(msg)
    if stderr:
        return stderr
    return stdout