from collections.abc import Sequence
from plotly import exceptions
from plotly.colors import (
def validate_index(index_vals):
    """
    Validates if a list contains all numbers or all strings

    :raises: (PlotlyError) If there are any two items in the list whose
        types differ
    """
    from numbers import Number
    if isinstance(index_vals[0], Number):
        if not all((isinstance(item, Number) for item in index_vals)):
            raise exceptions.PlotlyError('Error in indexing column. Make sure all entries of each column are all numbers or all strings.')
    elif isinstance(index_vals[0], str):
        if not all((isinstance(item, str) for item in index_vals)):
            raise exceptions.PlotlyError('Error in indexing column. Make sure all entries of each column are all numbers or all strings.')