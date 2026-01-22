import re
import traitlets
import datetime as dt
def date_to_json(pydate, manager):
    """Serialize a Python date object.

    Attributes of this dictionary are to be passed to the JavaScript Date
    constructor.
    """
    if pydate is None:
        return None
    else:
        return dict(year=pydate.year, month=pydate.month - 1, date=pydate.day)