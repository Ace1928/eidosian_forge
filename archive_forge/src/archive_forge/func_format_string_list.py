import re
from urllib import parse as urllib_parse
import pyparsing as pp
def format_string_list(objs, field):
    objs[field] = ', '.join(objs[field])