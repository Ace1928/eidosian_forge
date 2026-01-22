import sys
import logging
import os
import copy
from lib2to3.pgen2.parse import ParseError
from lib2to3.refactor import RefactoringTool
from libfuturize import fixes
def detect_python2(source, pathname):
    """
    Returns a bool indicating whether we think the code is Py2
    """
    RTs.setup_detect_python2()
    try:
        tree = RTs._rt_py2_detect.refactor_string(source, pathname)
    except ParseError as e:
        if e.msg != 'bad input' or e.value != '=':
            raise
        tree = RTs._rtp.refactor_string(source, pathname)
    if source != str(tree)[:-1]:
        logger.debug('Detected Python 2 code: {0}'.format(pathname))
        return True
    else:
        logger.debug('Detected Python 3 code: {0}'.format(pathname))
        return False