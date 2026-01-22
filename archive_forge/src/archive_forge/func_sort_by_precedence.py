import re
import itertools
import textwrap
import uuid
import param
from param.display import register_display_accessor
from param._utils import async_executor
def sort_by_precedence(self, parameters):
    """
        Sort the provided dictionary of parameters by their precedence value,
        preserving the original ordering for parameters with the
        same precedence.
        """
    params = [(p, pobj) for p, pobj in parameters.items()]
    key_fn = lambda x: x[1].precedence if x[1].precedence is not None else 1e-08
    sorted_params = sorted(params, key=key_fn)
    groups = itertools.groupby(sorted_params, key=key_fn)
    ordered_groups = [list(grp) for _, grp in groups]
    ordered_params = [el[0] for group in ordered_groups for el in group if el[0] != 'name' or el[0] in parameters]
    return ordered_params