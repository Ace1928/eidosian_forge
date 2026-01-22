from inspect import Parameter
from jedi.cache import memoize_method
from jedi import debug
from jedi import parser_utils
def param_strings():
    is_positional = False
    is_kw_only = False
    for n in self.get_param_names(resolve_stars=True):
        kind = n.get_kind()
        is_positional |= kind == Parameter.POSITIONAL_ONLY
        if is_positional and kind != Parameter.POSITIONAL_ONLY:
            yield '/'
            is_positional = False
        if kind == Parameter.VAR_POSITIONAL:
            is_kw_only = True
        elif kind == Parameter.KEYWORD_ONLY and (not is_kw_only):
            yield '*'
            is_kw_only = True
        yield n.to_string()
    if is_positional:
        yield '/'