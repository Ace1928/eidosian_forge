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
def ll_fz_subset_cff_for_gids(orig, num_gids, symbolic, cidfont):
    """
    Wrapper for out-params of fz_subset_cff_for_gids().
    Returns: fz_buffer *, int gids
    """
    outparams = ll_fz_subset_cff_for_gids_outparams()
    ret = ll_fz_subset_cff_for_gids_outparams_fn(orig, num_gids, symbolic, cidfont, outparams)
    return (ret, outparams.gids)