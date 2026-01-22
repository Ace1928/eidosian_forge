import os
import re
import random
from gimpfu import *
def v0_filename(w, h, pat, alpha, fmtinfo, testname, ext):
    return 'v0_{}x{}_{}_{:02X}_{}_{}_gimp.{}'.format(w, h, pat, alpha, fmtinfo, testname, ext)