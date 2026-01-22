from . import __version__
import copy
import re
import os
from .crackfortran import markoutercomma
from . import cb_rules
from ._isocbind import iso_c_binding_map, isoc_c2pycode_map, iso_c2py_map
from .auxfuncs import *
def cb_routsign2map(rout, um):
    """
    name,begintitle,endtitle,argname
    ctype,rctype,maxnofargs,nofoptargs,returncptr
    """
    ret = {'name': 'cb_%s_in_%s' % (rout['name'], um), 'returncptr': ''}
    if isintent_callback(rout):
        if '_' in rout['name']:
            F_FUNC = 'F_FUNC_US'
        else:
            F_FUNC = 'F_FUNC'
        ret['callbackname'] = '%s(%s,%s)' % (F_FUNC, rout['name'].lower(), rout['name'].upper())
        ret['static'] = 'extern'
    else:
        ret['callbackname'] = ret['name']
        ret['static'] = 'static'
    ret['argname'] = rout['name']
    ret['begintitle'] = gentitle(ret['name'])
    ret['endtitle'] = gentitle('end of %s' % ret['name'])
    ret['ctype'] = getctype(rout)
    ret['rctype'] = 'void'
    if ret['ctype'] == 'string':
        ret['rctype'] = 'void'
    else:
        ret['rctype'] = ret['ctype']
    if ret['rctype'] != 'void':
        if iscomplexfunction(rout):
            ret['returncptr'] = '\n#ifdef F2PY_CB_RETURNCOMPLEX\nreturn_value=\n#endif\n'
        else:
            ret['returncptr'] = 'return_value='
    if ret['ctype'] in cformat_map:
        ret['showvalueformat'] = '%s' % cformat_map[ret['ctype']]
    if isstringfunction(rout):
        ret['strlength'] = getstrlength(rout)
    if isfunction(rout):
        if 'result' in rout:
            a = rout['result']
        else:
            a = rout['name']
        if hasnote(rout['vars'][a]):
            ret['note'] = rout['vars'][a]['note']
            rout['vars'][a]['note'] = ['See elsewhere.']
        ret['rname'] = a
        ret['pydocsign'], ret['pydocsignout'] = getpydocsign(a, rout)
        if iscomplexfunction(rout):
            ret['rctype'] = '\n#ifdef F2PY_CB_RETURNCOMPLEX\n#ctype#\n#else\nvoid\n#endif\n'
    elif hasnote(rout):
        ret['note'] = rout['note']
        rout['note'] = ['See elsewhere.']
    nofargs = 0
    nofoptargs = 0
    if 'args' in rout and 'vars' in rout:
        for a in rout['args']:
            var = rout['vars'][a]
            if l_or(isintent_in, isintent_inout)(var):
                nofargs = nofargs + 1
                if isoptional(var):
                    nofoptargs = nofoptargs + 1
    ret['maxnofargs'] = repr(nofargs)
    ret['nofoptargs'] = repr(nofoptargs)
    if hasnote(rout) and isfunction(rout) and ('result' in rout):
        ret['routnote'] = rout['note']
        rout['note'] = ['See elsewhere.']
    return ret