from . import __version__
import copy
import re
import os
from .crackfortran import markoutercomma
from . import cb_rules
from ._isocbind import iso_c_binding_map, isoc_c2pycode_map, iso_c2py_map
from .auxfuncs import *
def modsign2map(m):
    """
    modulename
    """
    if ismodule(m):
        ret = {'f90modulename': m['name'], 'F90MODULENAME': m['name'].upper(), 'texf90modulename': m['name'].replace('_', '\\_')}
    else:
        ret = {'modulename': m['name'], 'MODULENAME': m['name'].upper(), 'texmodulename': m['name'].replace('_', '\\_')}
    ret['restdoc'] = getrestdoc(m) or []
    if hasnote(m):
        ret['note'] = m['note']
    ret['usercode'] = getusercode(m) or ''
    ret['usercode1'] = getusercode1(m) or ''
    if m['body']:
        ret['interface_usercode'] = getusercode(m['body'][0]) or ''
    else:
        ret['interface_usercode'] = ''
    ret['pymethoddef'] = getpymethoddef(m) or ''
    if 'coutput' in m:
        ret['coutput'] = m['coutput']
    if 'f2py_wrapper_output' in m:
        ret['f2py_wrapper_output'] = m['f2py_wrapper_output']
    return ret