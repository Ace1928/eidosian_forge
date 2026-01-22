import datetime
from functools import partial
import logging
def date_editor():
    """ Factory function that returns a Date editor for editing Date values.
    """
    from traitsui.api import DateEditor
    return DateEditor()