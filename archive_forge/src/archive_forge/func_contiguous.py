import operator
from OpenGL.arrays import buffers
from OpenGL.raw.GL import _types 
from OpenGL.raw.GL.VERSION import GL_1_1
from OpenGL import constant, error
@classmethod
def contiguous(cls, source, typeCode=None):
    """Get contiguous array from source

        source -- numpy Python array (or compatible object)
            for use as the data source.  If this is not a contiguous
            array of the given typeCode, a copy will be made,
            otherwise will just be returned unchanged.
        typeCode -- optional 1-character typeCode specifier for
            the numpy.array function.

        All gl*Pointer calls should use contiguous arrays, as non-
        contiguous arrays will be re-copied on every rendering pass.
        Although this doesn't raise an error, it does tend to slow
        down rendering.
        """
    typeCode = GL_TYPE_TO_ARRAY_MAPPING[typeCode]
    try:
        contiguous = source.flags.contiguous
    except AttributeError as err:
        if typeCode:
            return numpy.ascontiguousarray(source, typeCode)
        else:
            return numpy.ascontiguousarray(source)
    else:
        if contiguous and (typeCode is None or typeCode == source.dtype.char):
            return source
        elif contiguous and cls.ERROR_ON_COPY:
            from OpenGL import error
            raise error.CopyError('Array of type %r passed, required array of type %r', source.dtype.char, typeCode)
        else:
            if cls.ERROR_ON_COPY:
                from OpenGL import error
                raise error.CopyError('Non-contiguous array passed', source)
            if typeCode is None:
                typeCode = source.dtype.char
            return numpy.ascontiguousarray(source, typeCode)