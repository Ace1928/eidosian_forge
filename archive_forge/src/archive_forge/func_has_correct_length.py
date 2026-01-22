import os
import re
import sysconfig
def has_correct_length(length_range, start, end):
    """Determine if the line under test is within desired docstring length.

    This function is used with the --docstring-length min_rows max_rows
    argument.

    Parameters
    ----------
    length_range: list
        The file row range passed to the --docstring-length argument.
    start: int
        The row number where the line under test begins in the source file.
    end: int
        The row number where the line under tests ends in the source file.

    Returns
    -------
    correct_length: bool
        True if is correct length or length range is None, else False
    """
    if length_range is None:
        return True
    min_length, max_length = length_range
    docstring_length = end + 1 - start
    return min_length <= docstring_length <= max_length