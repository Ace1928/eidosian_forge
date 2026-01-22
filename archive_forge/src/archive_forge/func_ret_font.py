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
def ret_font(self, font):
    if font is None:
        return None
    elif isinstance(font, FzFont):
        return ll_fz_keep_font(font.m_internal)
    elif isinstance(font, fz_font):
        return font
    else:
        assert 0, f'Expected FzFont or fz_font, but fz_install_load_system_font_funcs() callback returned type(font)={type(font)!r}'