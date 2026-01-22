from __future__ import annotations
import os
import re
from typing import TYPE_CHECKING
import numpy as np
from monty.io import zopen
def preprocessor(data: str, dir: str='.') -> str:
    """
    Cp2k contains internal preprocessor flags that are evaluated before execution. This helper
    function recognizes those preprocessor flags and replaces them with an equivalent cp2k input
    (this way everything is contained neatly in the cp2k input structure, even if the user preferred
    to use the flags.

    CP2K preprocessor flags (with arguments) are:

        @INCLUDE FILENAME: Insert the contents of FILENAME into the file at
            this location.
        @SET VAR VALUE: set a variable, VAR, to have the value, VALUE.
        $VAR or ${VAR}: replace these with the value of the variable, as set
            by the @SET flag.
        @IF/@ELIF: Not implemented yet.

    Args:
        data (str): cp2k input to preprocess
        dir (str, optional): Path for include files. Default is '.' (current directory).

    Returns:
        Preprocessed string
    """
    includes = re.findall('(@include.+)', data, re.IGNORECASE)
    for incl in includes:
        inc = incl.split()
        assert len(inc) == 2
        inc = inc[1].strip("'")
        inc = inc.strip('"')
        with zopen(os.path.join(dir, inc)) as file:
            data = re.sub(f'{incl}', file.read(), data)
    variable_sets = re.findall('(@SET.+)', data, re.IGNORECASE)
    for match in variable_sets:
        v = match.split()
        assert len(v) == 3
        var, value = v[1:]
        data = re.sub(f'{match}', '', data)
        data = re.sub('\\${?' + var + '}?', value, data)
    c1 = re.findall('@IF', data, re.IGNORECASE)
    c2 = re.findall('@ELIF', data, re.IGNORECASE)
    if len(c1) > 0 or len(c2) > 0:
        raise NotImplementedError('This cp2k input processor does not currently support conditional blocks.')
    return data