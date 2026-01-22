from collections import OrderedDict
from ..chemistry import Substance
from .numbers import number_to_scientific_html
def map_fmt(cont, fmt, joiner='\n'):
    return joiner.join(map(lambda x: fmt % printer._print(x, **kwargs), cont))