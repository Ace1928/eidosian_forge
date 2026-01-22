import os
import subprocess
import sys
import re
from distutils.errors import DistutilsExecError, DistutilsPlatformError, \
from distutils.ccompiler import CCompiler, gen_lib_options
from distutils import log
from distutils.util import get_platform
import winreg
def find_vcvarsall(version):
    """Find the vcvarsall.bat file

    At first it tries to find the productdir of VS 2008 in the registry. If
    that fails it falls back to the VS90COMNTOOLS env var.
    """
    vsbase = VS_BASE % version
    try:
        productdir = Reg.get_value('%s\\Setup\\VC' % vsbase, 'productdir')
    except KeyError:
        log.debug('Unable to find productdir in registry')
        productdir = None
    if not productdir or not os.path.isdir(productdir):
        toolskey = 'VS%0.f0COMNTOOLS' % version
        toolsdir = os.environ.get(toolskey, None)
        if toolsdir and os.path.isdir(toolsdir):
            productdir = os.path.join(toolsdir, os.pardir, os.pardir, 'VC')
            productdir = os.path.abspath(productdir)
            if not os.path.isdir(productdir):
                log.debug('%s is not a valid directory' % productdir)
                return None
        else:
            log.debug('Env var %s is not set or invalid' % toolskey)
    if not productdir:
        log.debug('No productdir found')
        return None
    vcvarsall = os.path.join(productdir, 'vcvarsall.bat')
    if os.path.isfile(vcvarsall):
        return vcvarsall
    log.debug('Unable to find vcvarsall.bat')
    return None