import re
import textwrap
import param
from param.ipython import ParamPager
from param.parameterized import bothmethod
from .util import group_sanitizer, label_sanitizer
@classmethod
def get_parameter_info(cls, obj, ansi=False, show_values=True, pattern=None, max_col_len=40):
    """
        Get parameter information from the supplied class or object.
        """
    if cls.ppager is None:
        return ''
    if pattern is not None:
        obj = ParamFilter(obj, ParamFilter.regexp_filter(pattern))
        if len(list(obj.param)) <= 1:
            return None
    param_info = cls.ppager.get_param_info(obj)
    param_list = cls.ppager.param_docstrings(param_info)
    if not show_values:
        retval = cls.ansi_escape.sub('', param_list) if not ansi else param_list
        return cls.highlight(pattern, retval)
    else:
        info = cls.ppager(obj)
        if ansi is False:
            info = cls.ansi_escape.sub('', info)
        return cls.highlight(pattern, info)