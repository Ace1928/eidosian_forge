from nipype.interfaces import fsl
from nipype.utils import docparse
import subprocess
from ..interfaces.base import CommandLine
from .misc import is_container
def reverse_opt_map(opt_map):
    """Reverse the key/value pairs of the option map in the interface classes.

    Parameters
    ----------
    opt_map : dict
        Dictionary mapping the attribute name to a command line flag.
        Each interface class defines these for the command it wraps.

    Returns
    -------
    rev_opt_map : dict
       Dictionary mapping the flags to the attribute name.
    """
    revdict = {}
    for key, value in list(opt_map.items()):
        if is_container(value):
            value = value[0]
        if key != 'flags' and value is not None:
            revdict[value.split()[0]] = key
    return revdict