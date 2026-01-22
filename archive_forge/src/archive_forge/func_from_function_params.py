import re
import pydoc
from .external.docscrape import NumpyDocString
@classmethod
def from_function_params(cls, func):
    """Use the numpydoc parser to extract components from existing func."""
    params = NumpyDocString(pydoc.getdoc(func))['Parameters']
    comp_dict = {}
    for p in params:
        name = p.name
        type = p.type
        desc = '\n    '.join(p.desc)
        comp_dict[name] = f'{name} : {type}\n    {desc}'
    return cls(comp_dict)