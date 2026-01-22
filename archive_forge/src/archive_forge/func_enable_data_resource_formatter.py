from __future__ import annotations
from collections.abc import (
import sys
from typing import (
from unicodedata import east_asian_width
from pandas._config import get_option
from pandas.core.dtypes.inference import is_sequence
from pandas.io.formats.console import get_console_size
def enable_data_resource_formatter(enable: bool) -> None:
    if 'IPython' not in sys.modules:
        return
    from IPython import get_ipython
    ip = get_ipython()
    if ip is None:
        return
    formatters = ip.display_formatter.formatters
    mimetype = 'application/vnd.dataresource+json'
    if enable:
        if mimetype not in formatters:
            from IPython.core.formatters import BaseFormatter
            from traitlets import ObjectName

            class TableSchemaFormatter(BaseFormatter):
                print_method = ObjectName('_repr_data_resource_')
                _return_type = (dict,)
            formatters[mimetype] = TableSchemaFormatter()
        formatters[mimetype].enabled = True
    elif mimetype in formatters:
        formatters[mimetype].enabled = False