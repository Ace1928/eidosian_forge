from pyparsing import *
def typeAsCtypes(typestr):
    if typestr in typemap:
        return typemap[typestr]
    if typestr.endswith('*'):
        return 'POINTER(%s)' % typeAsCtypes(typestr.rstrip(' *'))
    return typestr