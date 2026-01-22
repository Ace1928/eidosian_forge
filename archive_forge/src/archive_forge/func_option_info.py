import re
import textwrap
import param
from param.ipython import ParamPager
from param.parameterized import bothmethod
from .util import group_sanitizer, label_sanitizer
@bothmethod
def option_info(cls_or_slf, node):
    if not cls_or_slf.show_options:
        return None
    from .options import Options, Store
    options = {}
    for g in Options._option_groups:
        gopts = Store.lookup_options(Store.current_backend, node, g, defaults=cls_or_slf.show_defaults)
        if gopts:
            options.update(gopts.kwargs)
    opts = Options(**{k: v for k, v in options.items() if k != 'backend'})
    return opts