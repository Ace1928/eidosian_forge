import pandas
from pandas.core.dtypes.common import is_list_like
from modin.core.dataframe.algebra.operator import Operator
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, try_cast_to_pandas
@classmethod
def build_property_wrapper(cls, prop):
    """
        Build function that accesses specified property of the frame.

        Parameters
        ----------
        prop : str
            Property name to access.

        Returns
        -------
        callable
            Function that takes DataFrame and returns its value of `prop` property.
        """

    def property_wrapper(df):
        """Get specified property of the passed object."""
        return prop.fget(df)
    return property_wrapper