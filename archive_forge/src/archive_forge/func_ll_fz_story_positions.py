from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def ll_fz_story_positions(story, python_callback):
    """
    Custom replacement for `ll_fz_story_positions()` that takes
    a Python callable `python_callback`.
    """
    python_callback_instance = StoryPositionsCallback_python(python_callback)
    ll_fz_story_positions_director(story, python_callback_instance)