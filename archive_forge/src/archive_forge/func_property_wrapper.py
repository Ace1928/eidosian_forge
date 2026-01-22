import pandas
from pandas.core.dtypes.common import is_list_like
from modin.core.dataframe.algebra.operator import Operator
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, try_cast_to_pandas
def property_wrapper(df):
    """Get specified property of the passed object."""
    return prop.fget(df)