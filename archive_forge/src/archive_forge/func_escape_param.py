from __future__ import absolute_import, division, print_function
import shlex
import pipes
import re
import json
import os
def escape_param(param):
    """
    Escapes the given parameter
    @param - The parameter to escape
    """
    escaped = None
    if hasattr(shlex, 'quote'):
        escaped = shlex.quote(param)
    elif hasattr(pipes, 'quote'):
        escaped = pipes.quote(param)
    else:
        escaped = "'" + param.replace("'", "'\\''") + "'"
    return escaped