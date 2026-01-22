import pandas
from pandas.core.dtypes.common import is_list_like
from modin.core.dataframe.algebra.operator import Operator
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, try_cast_to_pandas
@classmethod
def build_default_to_pandas(cls, fn, fn_name):
    """
        Build function that do fallback to pandas for passed `fn`.

        Parameters
        ----------
        fn : callable
            Function to apply to the defaulted frame.
        fn_name : str
            Function name which will be shown in default-to-pandas warning message.

        Returns
        -------
        callable
            Method that does fallback to pandas and applies `fn` to the pandas frame.
        """
    fn.__name__ = f'<function {cls.OBJECT_TYPE}.{fn_name}>'

    def wrapper(self, *args, **kwargs):
        """Do fallback to pandas for the specified function."""
        return self.default_to_pandas(fn, *args, **kwargs)
    return wrapper