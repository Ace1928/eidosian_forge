from __future__ import absolute_import
from .Visitor import CythonTransform
from .ModuleNode import ModuleNode
from .Errors import CompileError
from .UtilityCode import CythonUtilityCode
from .Code import UtilityCode, TempitaUtilityCode
from . import Options
from . import Interpreter
from . import PyrexTypes
from . import Naming
from . import Symtab
def get_type_information_cname(code, dtype, maxdepth=None):
    """
    Output the run-time type information (__Pyx_TypeInfo) for given dtype,
    and return the name of the type info struct.

    Structs with two floats of the same size are encoded as complex numbers.
    One can separate between complex numbers declared as struct or with native
    encoding by inspecting to see if the fields field of the type is
    filled in.
    """
    namesuffix = mangle_dtype_name(dtype)
    name = '__Pyx_TypeInfo_%s' % namesuffix
    structinfo_name = '__Pyx_StructFields_%s' % namesuffix
    if dtype.is_error:
        return '<error>'
    if maxdepth is None:
        maxdepth = dtype.struct_nesting_depth()
    if maxdepth <= 0:
        assert False
    if name not in code.globalstate.utility_codes:
        code.globalstate.utility_codes.add(name)
        typecode = code.globalstate['typeinfo']
        arraysizes = []
        if dtype.is_array:
            while dtype.is_array:
                arraysizes.append(dtype.size)
                dtype = dtype.base_type
        complex_possible = dtype.is_struct_or_union and dtype.can_be_complex()
        declcode = dtype.empty_declaration_code()
        if dtype.is_simple_buffer_dtype():
            structinfo_name = 'NULL'
        elif dtype.is_struct:
            struct_scope = dtype.scope
            if dtype.is_cv_qualified:
                struct_scope = struct_scope.base_type_scope
            fields = struct_scope.var_entries
            assert len(fields) > 0
            types = [get_type_information_cname(code, f.type, maxdepth - 1) for f in fields]
            typecode.putln('static __Pyx_StructField %s[] = {' % structinfo_name, safe=True)
            if dtype.is_cv_qualified:
                struct_type = dtype.cv_base_type.empty_declaration_code()
            else:
                struct_type = dtype.empty_declaration_code()
            for f, typeinfo in zip(fields, types):
                typecode.putln('  {&%s, "%s", offsetof(%s, %s)},' % (typeinfo, f.name, struct_type, f.cname), safe=True)
            typecode.putln('  {NULL, NULL, 0}', safe=True)
            typecode.putln('};', safe=True)
        else:
            assert False
        rep = str(dtype)
        flags = '0'
        is_unsigned = '0'
        if dtype is PyrexTypes.c_char_type:
            is_unsigned = '__PYX_IS_UNSIGNED(%s)' % declcode
            typegroup = "'H'"
        elif dtype.is_int:
            is_unsigned = '__PYX_IS_UNSIGNED(%s)' % declcode
            typegroup = "%s ? 'U' : 'I'" % is_unsigned
        elif complex_possible or dtype.is_complex:
            typegroup = "'C'"
        elif dtype.is_float:
            typegroup = "'R'"
        elif dtype.is_struct:
            typegroup = "'S'"
            if dtype.packed:
                flags = '__PYX_BUF_FLAGS_PACKED_STRUCT'
        elif dtype.is_pyobject:
            typegroup = "'O'"
        else:
            assert False, dtype
        typeinfo = 'static __Pyx_TypeInfo %s = { "%s", %s, sizeof(%s), { %s }, %s, %s, %s, %s };'
        tup = (name, rep, structinfo_name, declcode, ', '.join([str(x) for x in arraysizes]) or '0', len(arraysizes), typegroup, is_unsigned, flags)
        typecode.putln(typeinfo % tup, safe=True)
    return name