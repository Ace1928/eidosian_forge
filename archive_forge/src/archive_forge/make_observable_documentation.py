from __future__ import annotations
import numpy as np
Convert an observable in matrix form to dictionary form.

    Takes in a diagonal observable as a matrix and converts it to a dictionary
    form. Can also handle a list sorted of the diagonal elements.

    Args:
        matrix_observable (list): The observable to be converted to dictionary
        form. Can be a matrix or just an ordered list of observed values

    Returns:
        Dict: A dictionary with all observable states as keys, and corresponding
        values being the observed value for that state
    