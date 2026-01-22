from OpenGL.GL import *  # noqa
from OpenGL.GL import shaders  # noqa
import numpy as np
import re
def getShaderProgram(name):
    return ShaderProgram.names[name]