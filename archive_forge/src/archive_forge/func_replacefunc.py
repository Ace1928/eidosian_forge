import re
from pygments.token import String, Comment, Keyword, Name, Error, Whitespace, \
from pygments.filter import Filter
from pygments.util import get_list_opt, get_int_opt, get_bool_opt, \
from pygments.plugin import find_plugin_filters
def replacefunc(wschar):
    if wschar == ' ':
        return spaces
    elif wschar == '\t':
        return tabs
    elif wschar == '\n':
        return newlines
    return wschar