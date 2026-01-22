import re
from urllib import parse as urllib_parse
import pyparsing as pp
def format_archive_policy(ap):
    format_dict_list(ap, 'definition')
    format_string_list(ap, 'aggregation_methods')