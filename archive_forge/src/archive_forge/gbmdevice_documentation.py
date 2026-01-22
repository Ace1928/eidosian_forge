import weakref, ctypes, logging, os, glob
from OpenGL.platform import ctypesloader
from OpenGL import _opaque
Create a GBM surface to use on the given device
    
    devices -- opaque GBMDevice pointer
    width,height -- dimensions
    format -- EGL_NATIVE_VISUAL_ID from an EGL configuration
    flags -- surface flags regarding reading/writing pattern that
             is expected for the buffer
    
    returns GBMSurface opaque pointer
    