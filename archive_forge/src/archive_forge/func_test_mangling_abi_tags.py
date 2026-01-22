import unittest
from numba import njit
from numba.core.funcdesc import PythonFunctionDescriptor, default_mangler
from numba.core.compiler import run_frontend
from numba.core.itanium_mangler import mangle_abi_tag
def test_mangling_abi_tags(self):
    """
        This is a minimal test for the abi-tags support in the mangler.
        """

    def udt():
        pass
    func_ir = run_frontend(udt)
    typemap = {}
    restype = None
    calltypes = ()
    mangler = default_mangler
    inline = False
    noalias = False
    abi_tags = ('Shrubbery', 'Herring')
    fd = PythonFunctionDescriptor.from_specialized_function(func_ir, typemap, restype, calltypes, mangler, inline, noalias, abi_tags=abi_tags)
    self.assertIn(''.join([mangle_abi_tag(x) for x in abi_tags]), fd.mangled_name)