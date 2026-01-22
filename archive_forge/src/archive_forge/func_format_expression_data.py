import numbers
import prettytable
import yaml
from osc_lib import exceptions as exc
from oslo_serialization import jsonutils
def format_expression_data(data):
    string_list = list()
    for k, v in data.items():
        if k == 'dimensions':
            dim_str = format_dimensions(v)
            string_list.append(dim_str)
        else:
            if isinstance(v, numbers.Number):
                d_str = k + ': ' + str(v)
            else:
                d_str = k + ': ' + v
            string_list.append(d_str)
    return '\n'.join(string_list)