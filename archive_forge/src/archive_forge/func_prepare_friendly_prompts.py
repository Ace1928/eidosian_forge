import logging
import sys
from humanfriendly.compat import interactive_prompt
from humanfriendly.terminal import (
from humanfriendly.text import format, concatenate
def prepare_friendly_prompts():
    u"""
    Make interactive prompts more user friendly.

    The prompts presented by :func:`python2:raw_input()` (in Python 2) and
    :func:`python3:input()` (in Python 3) are not very user friendly by
    default, for example the cursor keys (:kbd:`←`, :kbd:`↑`, :kbd:`→` and
    :kbd:`↓`) and the :kbd:`Home` and :kbd:`End` keys enter characters instead
    of performing the action you would expect them to. By simply importing the
    :mod:`readline` module these prompts become much friendlier (as mentioned
    in the Python standard library documentation).

    This function is called by the other functions in this module to enable
    user friendly prompts.
    """
    try:
        import readline
    except ImportError:
        pass