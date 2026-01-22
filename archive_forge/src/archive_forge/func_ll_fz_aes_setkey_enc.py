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
def ll_fz_aes_setkey_enc(ctx, key, keysize):
    """
    Low-level wrapper for `::fz_aes_setkey_enc()`.
    AES encryption intialisation. Fills in the supplied context
    and prepares for encryption using the given key.

    Returns non-zero for error (key size other than 128/192/256).

    Never throws an exception.
    """
    return _mupdf.ll_fz_aes_setkey_enc(ctx, key, keysize)