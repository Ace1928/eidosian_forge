import os, sys, encodings
from reportlab.pdfbase import _fontdata
from reportlab.lib.logger import warnOnce
from reportlab.lib.utils import rl_isfile, rl_glob, rl_isdir, open_and_read, open_and_readlines, findInPaths, isSeq, isStr
from reportlab.rl_config import defaultEncoding, T1SearchPath
from reportlab.lib.rl_accel import unicode2T1, instanceStringWidthT1
from reportlab.pdfbase import rl_codecs
from reportlab.rl_config import register_reset
def bruteForceSearchForFile(fn, searchPath=None):
    if searchPath is None:
        from reportlab.rl_config import T1SearchPath as searchPath
    if rl_isfile(fn):
        return fn
    bfn = os.path.basename(fn)
    for dirname in searchPath:
        if not rl_isdir(dirname):
            continue
        tfn = os.path.join(dirname, bfn)
        if rl_isfile(tfn):
            return tfn
    return fn