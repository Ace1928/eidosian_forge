from . import hfk
import os
import json
import sys
def matches_saved_HFK_results():
    """
    >>> matches_saved_HFK_results()
    True
    """
    failures = 0
    for datum in regression_data():
        name = datum.pop('name')
        pd = datum.pop('PD_code')
        result = hfk.pd_to_hfk(repr(pd))
        if datum != result:
            print('Regression failure: ' + name)
            failures += 1
    return failures == 0