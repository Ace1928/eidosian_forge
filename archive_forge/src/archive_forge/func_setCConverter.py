import ctypes, logging
from OpenGL import platform, error
from OpenGL._configflags import STORE_POINTERS, ERROR_ON_COPY, SIZE_1_ARRAY_UNPACK
from OpenGL import converters
from OpenGL.converters import DefaultCConverter
from OpenGL.converters import returnCArgument,returnPyArgument
from OpenGL.latebind import LateBind
from OpenGL.arrays import arrayhelpers, arraydatatype
from OpenGL._null import NULL
from OpenGL import acceleratesupport
def setCConverter(self, argName, function):
    """Set C-argument converter for a given argument

        argName -- the argument name whose C-compatible representation will
            be calculated with the passed function.
        function -- None (indicating a simple copy), a non-callable object to
            be copied into the result-list itself, or a callable object with
            the signature:

                converter( pyArgs, index, wrappedOperation )

            where pyArgs is the set of passed Python arguments, with the
            pyConverters already applied, index is the index of the C argument
            and wrappedOperation is the underlying function.

        C-argument converters are your chance to expand/contract a Python
        argument list (pyArgs) to match the number of arguments expected by
        the ctypes baseOperation.  You can't have a "null" C-argument converter,
        as *something* has to be passed to the C-level function in the
        parameter.
        """
    if not hasattr(self, 'cConverters'):
        self.cConverters = [None] * len(self.wrappedOperation.argNames)
    try:
        if not isinstance(self.wrappedOperation.argNames, list):
            self.wrappedOperation.argNames = list(self.wrappedOperation.argNames)
        i = asList(self.wrappedOperation.argNames).index(argName)
    except ValueError:
        raise AttributeError('No argument named %r left in cConverters: %s' % (argName, self.wrappedOperation.argNames))
    if self.cConverters[i] is not None:
        raise RuntimeError('Double wrapping of output parameter: %r on %s' % (argName, self.__name__))
    self.cConverters[i] = function
    return self