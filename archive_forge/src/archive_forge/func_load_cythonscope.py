from __future__ import absolute_import
from .Symtab import ModuleScope
from .PyrexTypes import *
from .UtilityCode import CythonUtilityCode
from .Errors import error
from .Scanning import StringSourceDescriptor
from . import MemoryView
from .StringEncoding import EncodedString
def load_cythonscope(self):
    """
        Creates some entries for testing purposes and entries for
        cython.array() and for cython.view.*.
        """
    if self._cythonscope_initialized:
        return
    self._cythonscope_initialized = True
    cython_testscope_utility_code.declare_in_scope(self, cython_scope=self)
    cython_test_extclass_utility_code.declare_in_scope(self, cython_scope=self)
    self.viewscope = viewscope = ModuleScope(u'view', self, None)
    self.declare_module('view', viewscope, None).as_module = viewscope
    viewscope.is_cython_builtin = True
    viewscope.pxd_file_loaded = True
    cythonview_testscope_utility_code.declare_in_scope(viewscope, cython_scope=self)
    view_utility_scope = MemoryView.view_utility_code.declare_in_scope(self.viewscope, cython_scope=self, allowlist=MemoryView.view_utility_allowlist)
    ext_types = [entry.type for entry in view_utility_scope.entries.values() if entry.type.is_extension_type]
    for ext_type in ext_types:
        ext_type.is_cython_builtin_type = 1
    dc_str = EncodedString(u'dataclasses')
    dataclassesscope = ModuleScope(dc_str, self, context=None)
    self.declare_module(dc_str, dataclassesscope, pos=None).as_module = dataclassesscope
    dataclassesscope.is_cython_builtin = True
    dataclassesscope.pxd_file_loaded = True