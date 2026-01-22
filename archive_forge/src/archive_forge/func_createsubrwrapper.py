import copy
from .auxfuncs import (
from ._isocbind import isoc_kindmap
def createsubrwrapper(rout, signature=0):
    assert issubroutine(rout)
    extra_args = []
    vars = rout['vars']
    for a in rout['args']:
        v = rout['vars'][a]
        for i, d in enumerate(v.get('dimension', [])):
            if d == ':':
                dn = 'f2py_%s_d%s' % (a, i)
                dv = dict(typespec='integer', intent=['hide'])
                dv['='] = 'shape(%s, %s)' % (a, i)
                extra_args.append(dn)
                vars[dn] = dv
                v['dimension'][i] = dn
    rout['args'].extend(extra_args)
    need_interface = bool(extra_args)
    ret = ['']

    def add(line, ret=ret):
        ret[0] = '%s\n      %s' % (ret[0], line)
    name = rout['name']
    fortranname = getfortranname(rout)
    f90mode = ismoduleroutine(rout)
    args = rout['args']
    useisoc = useiso_c_binding(rout)
    sargs = ', '.join(args)
    if f90mode:
        add('subroutine f2pywrap_%s_%s (%s)' % (rout['modulename'], name, sargs))
        if useisoc:
            add('use iso_c_binding')
        if not signature:
            add('use %s, only : %s' % (rout['modulename'], fortranname))
    else:
        add('subroutine f2pywrap%s (%s)' % (name, sargs))
        if useisoc:
            add('use iso_c_binding')
        if not need_interface:
            add('external %s' % fortranname)
    if need_interface:
        for line in rout['saved_interface'].split('\n'):
            if line.lstrip().startswith('use ') and '__user__' not in line:
                add(line)
    dumped_args = []
    for a in args:
        if isexternal(vars[a]):
            add('external %s' % a)
            dumped_args.append(a)
    for a in args:
        if a in dumped_args:
            continue
        if isscalar(vars[a]):
            add(var2fixfortran(vars, a, f90mode=f90mode))
            dumped_args.append(a)
    for a in args:
        if a in dumped_args:
            continue
        add(var2fixfortran(vars, a, f90mode=f90mode))
    if need_interface:
        if f90mode:
            pass
        else:
            add('interface')
            for line in rout['saved_interface'].split('\n'):
                if line.lstrip().startswith('use ') and '__user__' in line:
                    continue
                add(line)
            add('end interface')
    sargs = ', '.join([a for a in args if a not in extra_args])
    if not signature:
        add('call %s(%s)' % (fortranname, sargs))
    if f90mode:
        add('end subroutine f2pywrap_%s_%s' % (rout['modulename'], name))
    else:
        add('end')
    return ret[0]