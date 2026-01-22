import ctypes
from OpenGL.platform import ctypesloader
from OpenGL._bytes import as_8_bit
import sys, logging
from OpenGL import _configflags
from OpenGL import logs, MODULE_ANNOTATIONS
def constructFunction(self, functionName, dll, resultType=ctypes.c_int, argTypes=(), doc=None, argNames=(), extension=None, deprecated=False, module=None, force_extension=False, error_checker=None):
    """Core operation to create a new base ctypes function
        
        raises AttributeError if can't find the procedure...
        """
    is_core = not extension or extension.split('_')[1] == 'VERSION'
    if not is_core and (not self.checkExtension(extension)):
        raise AttributeError('Extension not available')
    argTypes = [self.finalArgType(t) for t in argTypes]
    if force_extension or (not is_core and (not self.EXTENSIONS_USE_BASE_FUNCTIONS)):
        pointer = self.getExtensionProcedure(as_8_bit(functionName))
        if pointer:
            func = self.functionTypeFor(dll)(resultType, *argTypes)(pointer)
        else:
            raise AttributeError('Extension %r available, but no pointer for function %r' % (extension, functionName))
    else:
        func = ctypesloader.buildFunction(self.functionTypeFor(dll)(resultType, *argTypes), functionName, dll)
    func.__doc__ = doc
    func.argNames = list(argNames or ())
    func.__name__ = functionName
    func.DLL = dll
    func.extension = extension
    func.deprecated = deprecated
    func = self.wrapLogging(self.wrapContextCheck(self.errorChecking(func, dll, error_checker=error_checker), dll))
    if MODULE_ANNOTATIONS:
        if not module:
            module = _find_module()
        if module:
            func.__module__ = module
    return func