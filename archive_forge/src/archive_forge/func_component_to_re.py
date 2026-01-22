import re
from typing import Iterable, Union
def component_to_re(component):
    if '**' in component:
        if component == '**':
            return '(' + re.escape(separator) + '[^' + separator + ']+)*'
        else:
            raise ValueError('** can only appear as an entire path segment')
    else:
        return re.escape(separator) + ('[^' + separator + ']*').join((re.escape(x) for x in component.split('*')))