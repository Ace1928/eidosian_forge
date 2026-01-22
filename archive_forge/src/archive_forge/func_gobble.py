import re
from pygments.token import String, Comment, Keyword, Name, Error, Whitespace, \
from pygments.filter import Filter
from pygments.util import get_list_opt, get_int_opt, get_bool_opt, \
from pygments.plugin import find_plugin_filters
def gobble(self, value, left):
    if left < len(value):
        return (value[left:], 0)
    else:
        return (u'', left - len(value))