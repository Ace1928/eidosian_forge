import os, sys, io
from . import ffiplatform, model
from .error import VerificationError
from .cffi_opcode import *
def write_c_source_to_f(self, f, preamble):
    self._f = f
    prnt = self._prnt
    if self.ffi._embedding is not None:
        prnt('#define _CFFI_USE_EMBEDDING')
    if not USE_LIMITED_API:
        prnt('#define _CFFI_NO_LIMITED_API')
    lines = self._rel_readlines('_cffi_include.h')
    i = lines.index('#include "parse_c_type.h"\n')
    lines[i:i + 1] = self._rel_readlines('parse_c_type.h')
    prnt(''.join(lines))
    base_module_name = self.module_name.split('.')[-1]
    if self.ffi._embedding is not None:
        prnt('#define _CFFI_MODULE_NAME  "%s"' % (self.module_name,))
        prnt('static const char _CFFI_PYTHON_STARTUP_CODE[] = {')
        self._print_string_literal_in_array(self.ffi._embedding)
        prnt('0 };')
        prnt('#ifdef PYPY_VERSION')
        prnt('# define _CFFI_PYTHON_STARTUP_FUNC  _cffi_pypyinit_%s' % (base_module_name,))
        prnt('#elif PY_MAJOR_VERSION >= 3')
        prnt('# define _CFFI_PYTHON_STARTUP_FUNC  PyInit_%s' % (base_module_name,))
        prnt('#else')
        prnt('# define _CFFI_PYTHON_STARTUP_FUNC  init%s' % (base_module_name,))
        prnt('#endif')
        lines = self._rel_readlines('_embedding.h')
        i = lines.index('#include "_cffi_errors.h"\n')
        lines[i:i + 1] = self._rel_readlines('_cffi_errors.h')
        prnt(''.join(lines))
        self.needs_version(VERSION_EMBEDDED)
    prnt('/************************************************************/')
    prnt()
    prnt(preamble)
    prnt()
    prnt('/************************************************************/')
    prnt()
    prnt('static void *_cffi_types[] = {')
    typeindex2type = dict([(i, tp) for tp, i in self._typesdict.items()])
    for i, op in enumerate(self.cffi_types):
        comment = ''
        if i in typeindex2type:
            comment = ' // ' + typeindex2type[i]._get_c_name()
        prnt('/* %2d */ %s,%s' % (i, op.as_c_expr(), comment))
    if not self.cffi_types:
        prnt('  0')
    prnt('};')
    prnt()
    self._seen_constants = set()
    self._generate('decl')
    nums = {}
    for step_name in self.ALL_STEPS:
        lst = self._lsts[step_name]
        nums[step_name] = len(lst)
        if nums[step_name] > 0:
            prnt('static const struct _cffi_%s_s _cffi_%ss[] = {' % (step_name, step_name))
            for entry in lst:
                prnt(entry.as_c_expr())
            prnt('};')
            prnt()
    if self.ffi._included_ffis:
        prnt('static const char * const _cffi_includes[] = {')
        for ffi_to_include in self.ffi._included_ffis:
            try:
                included_module_name, included_source = ffi_to_include._assigned_source[:2]
            except AttributeError:
                raise VerificationError('ffi object %r includes %r, but the latter has not been prepared with set_source()' % (self.ffi, ffi_to_include))
            if included_source is None:
                raise VerificationError('not implemented yet: ffi.include() of a Python-based ffi inside a C-based ffi')
            prnt('  "%s",' % (included_module_name,))
        prnt('  NULL')
        prnt('};')
        prnt()
    prnt('static const struct _cffi_type_context_s _cffi_type_context = {')
    prnt('  _cffi_types,')
    for step_name in self.ALL_STEPS:
        if nums[step_name] > 0:
            prnt('  _cffi_%ss,' % step_name)
        else:
            prnt('  NULL,  /* no %ss */' % step_name)
    for step_name in self.ALL_STEPS:
        if step_name != 'field':
            prnt('  %d,  /* num_%ss */' % (nums[step_name], step_name))
    if self.ffi._included_ffis:
        prnt('  _cffi_includes,')
    else:
        prnt('  NULL,  /* no includes */')
    prnt('  %d,  /* num_types */' % (len(self.cffi_types),))
    flags = 0
    if self._num_externpy > 0 or self.ffi._embedding is not None:
        flags |= 1
    prnt('  %d,  /* flags */' % flags)
    prnt('};')
    prnt()
    prnt('#ifdef __GNUC__')
    prnt('#  pragma GCC visibility push(default)  /* for -fvisibility= */')
    prnt('#endif')
    prnt()
    prnt('#ifdef PYPY_VERSION')
    prnt('PyMODINIT_FUNC')
    prnt('_cffi_pypyinit_%s(const void *p[])' % (base_module_name,))
    prnt('{')
    if flags & 1:
        prnt('    if (((intptr_t)p[0]) >= 0x0A03) {')
        prnt('        _cffi_call_python_org = (void(*)(struct _cffi_externpy_s *, char *))p[1];')
        prnt('    }')
    prnt('    p[0] = (const void *)0x%x;' % self._version)
    prnt('    p[1] = &_cffi_type_context;')
    prnt('#if PY_MAJOR_VERSION >= 3')
    prnt('    return NULL;')
    prnt('#endif')
    prnt('}')
    prnt('#  ifdef _MSC_VER')
    prnt('     PyMODINIT_FUNC')
    prnt('#  if PY_MAJOR_VERSION >= 3')
    prnt('     PyInit_%s(void) { return NULL; }' % (base_module_name,))
    prnt('#  else')
    prnt('     init%s(void) { }' % (base_module_name,))
    prnt('#  endif')
    prnt('#  endif')
    prnt('#elif PY_MAJOR_VERSION >= 3')
    prnt('PyMODINIT_FUNC')
    prnt('PyInit_%s(void)' % (base_module_name,))
    prnt('{')
    prnt('  return _cffi_init("%s", 0x%x, &_cffi_type_context);' % (self.module_name, self._version))
    prnt('}')
    prnt('#else')
    prnt('PyMODINIT_FUNC')
    prnt('init%s(void)' % (base_module_name,))
    prnt('{')
    prnt('  _cffi_init("%s", 0x%x, &_cffi_type_context);' % (self.module_name, self._version))
    prnt('}')
    prnt('#endif')
    prnt()
    prnt('#ifdef __GNUC__')
    prnt('#  pragma GCC visibility pop')
    prnt('#endif')
    self._version = None