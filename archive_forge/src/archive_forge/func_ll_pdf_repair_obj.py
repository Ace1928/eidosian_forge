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
def ll_pdf_repair_obj(doc, buf):
    """
    Wrapper for out-params of pdf_repair_obj().
    Returns: int, int64_t stmofsp, int64_t stmlenp, ::pdf_obj *encrypt, ::pdf_obj *id, ::pdf_obj *page, int64_t tmpofs, ::pdf_obj *root
    """
    outparams = ll_pdf_repair_obj_outparams()
    ret = ll_pdf_repair_obj_outparams_fn(doc, buf, outparams)
    return (ret, outparams.stmofsp, outparams.stmlenp, outparams.encrypt, outparams.id, outparams.page, outparams.tmpofs, outparams.root)