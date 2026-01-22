import re
import sys
from docutils import DataError
from docutils.utils import strip_combining_chars
def update_dict_of_lists(master, newdata):
    """
    Extend the list values of `master` with those from `newdata`.

    Both parameters must be dictionaries containing list values.
    """
    for key, values in list(newdata.items()):
        master.setdefault(key, []).extend(values)