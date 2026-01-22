import pandas
from pandas.core.dtypes.common import is_list_like
from .default import DefaultMethod
def bin_ops_wrapper(df, other, *args, **kwargs):
    """Apply specified binary function to the passed operands."""
    squeeze_other = kwargs.pop('broadcast', False) or kwargs.pop('squeeze_other', False)
    squeeze_self = kwargs.pop('squeeze_self', False)
    if squeeze_other:
        other = other.squeeze(axis=1)
    if squeeze_self:
        df = df.squeeze(axis=1)
    result = fn(df, other, *args, **kwargs)
    if not isinstance(result, pandas.Series) and (not isinstance(result, pandas.DataFrame)) and is_list_like(result):
        result = pandas.DataFrame(result)
    return result