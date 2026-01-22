from OpenGL.GL import *  # noqa
from OpenGL import GL
from .. import Transform3D
from ..Qt import QtCore
def setGLOptions(self, opts):
    """
        Set the OpenGL state options to use immediately before drawing this item.
        (Note that subclasses must call setupGLState before painting for this to work)
        
        The simplest way to invoke this method is to pass in the name of
        a predefined set of options (see the GLOptions variable):
        
        ============= ======================================================
        opaque        Enables depth testing and disables blending
        translucent   Enables depth testing and blending
                      Elements must be drawn sorted back-to-front for
                      translucency to work correctly.
        additive      Disables depth testing, enables blending.
                      Colors are added together, so sorting is not required.
        ============= ======================================================
        
        It is also possible to specify any arbitrary settings as a dictionary. 
        This may consist of {'functionName': (args...)} pairs where functionName must 
        be a callable attribute of OpenGL.GL, or {GL_STATE_VAR: bool} pairs 
        which will be interpreted as calls to glEnable or glDisable(GL_STATE_VAR).
        
        For example::
            
            {
                GL_ALPHA_TEST: True,
                GL_CULL_FACE: False,
                'glBlendFunc': (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA),
            }
            
        
        """
    if isinstance(opts, str):
        opts = GLOptions[opts]
    self.__glOpts = opts.copy()
    self.update()