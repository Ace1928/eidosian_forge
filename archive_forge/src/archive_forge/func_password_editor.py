import datetime
from functools import partial
import logging
def password_editor(auto_set=True, enter_set=False):
    """ Factory function that returns an editor for passwords.
    """
    from traitsui.api import TextEditor
    return TextEditor(password=True, auto_set=auto_set, enter_set=enter_set)