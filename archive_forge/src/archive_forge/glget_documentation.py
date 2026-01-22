from OpenGL.GL.VERSION import GL_1_1 as _simple
import ctypes
Implementation of the special "glGet" functions

For comparison, here's what a straightforward implementation looks like:

    def glGetDoublev( pname ):
        "Natural writing of glGetDoublev using standard ctypes"
        output = c_double*sizes.get( pname )
        result = output()
        result = platform.PLATFORM.GL.glGetDoublev( pname, byref(result) )
        return Numeric.array( result )
