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
def setPyConverter(self, argName, function=NULL):
    """Set Python-argument converter for given argument

        argName -- the argument name which will be coerced to a usable internal
            format using the function provided.
        function -- None (indicating a simple copy), NULL (default) to eliminate
            the argument from the Python argument-list, or a callable object with
            the signature:

                converter(arg, wrappedOperation, args)

            where arg is the particular argument on which the convert is working,
            wrappedOperation is the underlying wrapper, and args is the set of
            original Python arguments to the function.

        Note that you need exactly the same number of pyConverters as Python
        arguments.
        """
    if not hasattr(self, 'pyConverters'):
        self.pyConverters = [None] * len(self.wrappedOperation.argNames)
        self.pyConverterNames = list(self.wrappedOperation.argNames)
    try:
        i = asList(self.pyConverterNames).index(argName)
    except ValueError:
        raise AttributeError('No argument named %r left in pyConverters for %r: %s' % (argName, self.wrappedOperation.__name__, self.pyConverterNames))
    if function is NULL:
        del self.pyConverters[i]
        del self.pyConverterNames[i]
    else:
        self.pyConverters[i] = function
    return self