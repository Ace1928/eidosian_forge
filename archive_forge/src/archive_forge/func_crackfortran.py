import sys
import string
import fileinput
import re
import os
import copy
import platform
import codecs
from pathlib import Path
from . import __version__
from .auxfuncs import *
from . import symbolic
def crackfortran(files):
    global usermodules, post_processing_hooks
    outmess('Reading fortran codes...\n', 0)
    readfortrancode(files, crackline)
    outmess('Post-processing...\n', 0)
    usermodules = []
    postlist = postcrack(grouplist[0])
    outmess('Applying post-processing hooks...\n', 0)
    for hook in post_processing_hooks:
        outmess(f'  {hook.__name__}\n', 0)
        postlist = traverse(postlist, hook)
    outmess('Post-processing (stage 2)...\n', 0)
    postlist = postcrack2(postlist)
    return usermodules + postlist