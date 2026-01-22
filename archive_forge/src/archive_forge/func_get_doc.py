from nipype.interfaces import fsl
from nipype.utils import docparse
import subprocess
from ..interfaces.base import CommandLine
from .misc import is_container
def get_doc(cmd, opt_map, help_flag=None, trap_error=True):
    """Get the docstring from our command and options map.

    Parameters
    ----------
    cmd : string
        The command whose documentation we are fetching
    opt_map : dict
        Dictionary of flags and option attributes.
    help_flag : string
        Provide additional help flag. e.g., -h
    trap_error : boolean
        Override if underlying command returns a non-zero returncode

    Returns
    -------
    doc : string
        The formatted docstring

    """
    res = CommandLine('which %s' % cmd.split(' ')[0], resource_monitor=False, terminal_output='allatonce').run()
    cmd_path = res.runtime.stdout.strip()
    if cmd_path == '':
        raise Exception('Command %s not found' % cmd.split(' ')[0])
    if help_flag:
        cmd = ' '.join((cmd, help_flag))
    doc = grab_doc(cmd, trap_error)
    opts = reverse_opt_map(opt_map)
    return build_doc(doc, opts)