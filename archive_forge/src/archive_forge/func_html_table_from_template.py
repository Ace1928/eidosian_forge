import json
import logging
import re
import uuid
import warnings
from base64 import b64encode
from pathlib import Path
import numpy as np
import pandas as pd
from .utils import UNPKG_DT_BUNDLE_CSS, UNPKG_DT_BUNDLE_URL
from .version import __version__ as itables_version
from IPython.display import HTML, display
import itables.options as opt
from .datatables_format import datatables_rows
from .downsample import downsample
from .utils import read_package_file
def html_table_from_template(html_table, table_id, data, kwargs, connected, column_filters):
    if 'css' in kwargs:
        TypeError("The 'css' argument has been deprecated, see the new approach at https://mwouts.github.io/itables/custom_css.html.")
    eval_functions = kwargs.pop('eval_functions', None)
    pre_dt_code = kwargs.pop('pre_dt_code')
    dt_url = kwargs.pop('dt_url')
    output = read_package_file('html/datatables_template.html')
    if connected:
        assert dt_url.endswith('.js')
        output = replace_value(output, UNPKG_DT_BUNDLE_URL, dt_url)
        output = replace_value(output, UNPKG_DT_BUNDLE_CSS, dt_url[:-3] + '.css')
    else:
        connected_style = f'<link href="{UNPKG_DT_BUNDLE_CSS}" rel="stylesheet">\n'
        output = replace_value(output, connected_style, '')
        connected_import = "import {DataTable, jQuery as $} from '" + UNPKG_DT_BUNDLE_URL + "';"
        local_import = 'const { DataTable, jQuery: $ } = await import(window.' + DATATABLES_SRC_FOR_ITABLES + ');'
        output = replace_value(output, connected_import, local_import)
    output = replace_value(output, '<table id="table_id"><thead><tr><th>A</th></tr></thead></table>', html_table)
    output = replace_value(output, '#table_id', '#{}'.format(table_id))
    if column_filters:
        assert pre_dt_code == ''
        assert 'initComplete' not in kwargs
        pre_dt_code = replace_value(read_package_file('html/column_filters/pre_dt_code.js'), 'thead_or_tfoot', 'thead' if column_filters == 'header' else 'tfoot')
        kwargs['initComplete'] = JavascriptFunction(replace_value(replace_value(read_package_file('html/column_filters/initComplete.js'), 'const initComplete = ', ''), 'header', column_filters))
    dt_args = json_dumps(kwargs, eval_functions)
    output = replace_value(output, 'let dt_args = {};', 'let dt_args = {};'.format(dt_args))
    output = replace_value(output, '// [pre-dt-code]', pre_dt_code.replace('#table_id', '#{}'.format(table_id)))
    if data is not None:
        output = replace_value(output, 'const data = [];', 'const data = {};'.format(data))
    else:
        output = replace_value(output, 'dt_args["data"] = data;', '')
        output = replace_value(output, 'const data = [];', '')
    return output