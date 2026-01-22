from __future__ import annotations
from textwrap import dedent
from typing import (
from pandas._config import get_option
from pandas._libs import lib
from pandas import (
from pandas.io.common import is_url
from pandas.io.formats.format import (
from pandas.io.formats.printing import pprint_thing
def write_style(self) -> None:
    template_first = '            <style scoped>'
    template_last = '            </style>'
    template_select = '                .dataframe %s {\n                    %s: %s;\n                }'
    element_props = [('tbody tr th:only-of-type', 'vertical-align', 'middle'), ('tbody tr th', 'vertical-align', 'top')]
    if isinstance(self.columns, MultiIndex):
        element_props.append(('thead tr th', 'text-align', 'left'))
        if self.show_row_idx_names:
            element_props.append(('thead tr:last-of-type th', 'text-align', 'right'))
    else:
        element_props.append(('thead th', 'text-align', 'right'))
    template_mid = '\n\n'.join((template_select % t for t in element_props))
    template = dedent(f'{template_first}\n{template_mid}\n{template_last}')
    self.write(template)