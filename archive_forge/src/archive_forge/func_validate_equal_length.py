from collections.abc import Sequence
from plotly import exceptions
from plotly.colors import (
def validate_equal_length(*args):
    """
    Validates that data lists or ndarrays are the same length.

    :raises: (PlotlyError) If any data lists are not the same length.
    """
    length = len(args[0])
    if any((len(lst) != length for lst in args)):
        raise exceptions.PlotlyError('Oops! Your data lists or ndarrays should be the same length.')