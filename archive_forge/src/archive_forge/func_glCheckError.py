import logging
from OpenGL import platform, _configflags
from ctypes import ArgumentError
def glCheckError(self, result, baseOperation=None, cArguments=None, *args):
    """Base GL Error checker compatible with new ctypes errcheck protocol
                
                This function will raise a GLError with just the calling information
                available at the C-calling level, i.e. the error code, cArguments,
                baseOperation and result.  Higher-level code is responsible for any 
                extra annotations.
                
                Note:
                    glCheckError relies on glBegin/glEnd interactions to 
                    prevent glGetError being called during a glBegin/glEnd 
                    sequence.  If you are calling glBegin/glEnd in C you 
                    should call onBegin and onEnd appropriately.
                """
    err = self._currentChecker()
    if err != self._noErrorResult:
        raise self._errorClass(err, result, cArguments=cArguments, baseOperation=baseOperation)
    return result