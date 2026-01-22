import re
import time
import platform
from collections import OrderedDict
import six
def get_keyboard_codes():
    """
    Return mapping of keycode integer values paired by their curses key-name.

    :rtype: dict
    :returns: Dictionary of (code, name) pairs for curses keyboard constant
        values and their mnemonic name. Such as key ``260``, with the value of
        its identity, ``u'KEY_LEFT'``.

    These keys are derived from the attributes by the same of the curses module,
    with the following exceptions:

    * ``KEY_DELETE`` in place of ``KEY_DC``
    * ``KEY_INSERT`` in place of ``KEY_IC``
    * ``KEY_PGUP`` in place of ``KEY_PPAGE``
    * ``KEY_PGDOWN`` in place of ``KEY_NPAGE``
    * ``KEY_ESCAPE`` in place of ``KEY_EXIT``
    * ``KEY_SUP`` in place of ``KEY_SR``
    * ``KEY_SDOWN`` in place of ``KEY_SF``

    This function is the inverse of :func:`get_curses_keycodes`.  With the
    given override "mixins" listed above, the keycode for the delete key will
    map to our imaginary ``KEY_DELETE`` mnemonic, effectively erasing the
    phrase ``KEY_DC`` from our code vocabulary for anyone that wishes to use
    the return value to determine the key-name by keycode.
    """
    keycodes = OrderedDict(get_curses_keycodes())
    keycodes.update(CURSES_KEYCODE_OVERRIDE_MIXIN)
    keycodes.update(((name, value) for name, value in globals().copy().items() if name.startswith('KEY_')))
    return dict(zip(keycodes.values(), keycodes.keys()))