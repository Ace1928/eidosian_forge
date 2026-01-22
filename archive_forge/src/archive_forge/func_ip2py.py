import doctest
import logging
import re
from testpath import modified_env
def ip2py(self, source):
    """Convert input IPython source into valid Python."""
    block = _ip.input_transformer_manager.transform_cell(source)
    if len(block.splitlines()) == 1:
        return _ip.prefilter(block)
    else:
        return block