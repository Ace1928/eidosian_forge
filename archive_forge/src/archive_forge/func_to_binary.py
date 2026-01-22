from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
def to_binary(self, *varargs, **keys):
    """data = s.to_binary(...)

        Convert Python values into the binary representation.  The
        arguments will be all value fields with names, in the order
        given when the Struct object was instantiated.  With one
        exception: fields with default arguments will be last.

        Returns the binary representation as the string DATA.

        """
    code = ''
    total_length = str(self.static_size)
    joins = []
    args = []
    defargs = []
    kwarg = 0
    i = 0
    for f in self.var_fields:
        if f.keyword_args:
            kwarg = 1
            kw = ', _keyword_args'
        else:
            kw = ''
        code = code + '  _%(name)s, _%(name)s_length, _%(name)s_format = self.var_fields[%(fno)d].pack_value(%(name)s%(kw)s)\n' % {'name': f.name, 'fno': i, 'kw': kw}
        total_length = total_length + ' + len(_%s)' % f.name
        joins.append('_%s' % f.name)
        i = i + 1
    pack_args = ['"%s"' % self.static_codes]
    i = 0
    for f in self.static_fields:
        if isinstance(f, LengthField):
            if isinstance(f, TotalLengthField):
                if self.var_fields:
                    pack_args.append('self.static_fields[%d].calc_length(%s)' % (i, total_length))
                else:
                    pack_args.append(str(f.calc_length(self.static_size)))
            else:
                pack_args.append('self.static_fields[%d].calc_length(_%s_length)' % (i, f.name))
        elif isinstance(f, FormatField):
            pack_args.append('_%s_format' % f.name)
        elif isinstance(f, ConstantField):
            pack_args.append(str(f.value))
        else:
            if f.structvalues == 1:
                if f.check_value is not None:
                    pack_args.append('self.static_fields[%d].check_value(%s)' % (i, f.name))
                else:
                    pack_args.append(f.name)
            else:
                a = []
                for j in range(f.structvalues):
                    a.append('_%s_%d' % (f.name, j))
                if f.check_value is not None:
                    code = code + '  %s = self.static_fields[%d].check_value(%s)\n' % (', '.join(a), i, f.name)
                else:
                    code = code + '  %s = %s\n' % (', '.join(a), f.name)
                pack_args = pack_args + a
            if f.name:
                if f.default is None:
                    args.append(f.name)
                else:
                    defargs.append('%s = %s' % (f.name, repr(f.default)))
        i = i + 1
    pack = 'struct.pack(%s)' % ', '.join(pack_args)
    if self.var_fields:
        code = code + '  return %s + %s\n' % (pack, ' + '.join(joins))
    else:
        code = code + '  return %s\n' % pack
    for f in self.var_fields:
        if f.name:
            if f.default is None:
                args.append(f.name)
            else:
                defargs.append('%s = %s' % (f.name, repr(f.default)))
    args = args + defargs
    if kwarg:
        args.append('**_keyword_args')
    code = 'def to_binary(self, %s):\n' % ', '.join(args) + code
    g = globals().copy()
    exec(code, g)
    self.to_binary = types.MethodType(g['to_binary'], self)
    return self.to_binary(*varargs, **keys)