import types
from _pydevd_bundle.pydevd_constants import IS_JYTHON
from _pydev_bundle._pydev_imports_tipper import signature_from_docstring
def create_function_stub(fn_name, fn_argspec, fn_docstring, indent=0):

    def shift_right(string, prefix):
        return ''.join((prefix + line for line in string.splitlines(True)))
    fn_docstring = shift_right(inspect.cleandoc(fn_docstring), '  ' * (indent + 1))
    ret = '\ndef %s%s:\n    """%s"""\n    pass\n' % (fn_name, fn_argspec, fn_docstring)
    ret = ret[1:]
    ret = ret.replace('\t', '  ')
    if indent:
        prefix = '  ' * indent
        ret = shift_right(ret, prefix)
    return ret