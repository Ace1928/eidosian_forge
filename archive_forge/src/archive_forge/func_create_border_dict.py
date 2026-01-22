import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.css import CSSResolver
def create_border_dict(sides, color=None, style=None, width=None):
    resolved = {}
    for side in sides:
        if color:
            resolved[f'border-{side}-color'] = color
        if style:
            resolved[f'border-{side}-style'] = style
        if width:
            resolved[f'border-{side}-width'] = width
    return resolved