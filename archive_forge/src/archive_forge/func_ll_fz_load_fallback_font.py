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
def ll_fz_load_fallback_font(script, language, serif, bold, italic):
    """
    Low-level wrapper for `::fz_load_fallback_font()`.
    Try to load a fallback font for the
    given combination of font attributes. Whether a font is
    present or not will depend on the configuration in which
    MuPDF is built.

    script: The script desired (e.g. UCDN_SCRIPT_KATAKANA).

    language: The language desired (e.g. FZ_LANG_ja).

    serif: 1 if serif desired, 0 otherwise.

    bold: 1 if bold desired, 0 otherwise.

    italic: 1 if italic desired, 0 otherwise.

    Returns a new font handle, or NULL if not available.
    """
    return _mupdf.ll_fz_load_fallback_font(script, language, serif, bold, italic)