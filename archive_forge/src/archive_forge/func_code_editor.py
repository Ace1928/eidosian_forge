import datetime
from functools import partial
import logging
def code_editor():
    """ Factory function that returns an editor that treats a multi-line string
    as source code.
    """
    from traitsui.api import CodeEditor
    return CodeEditor()