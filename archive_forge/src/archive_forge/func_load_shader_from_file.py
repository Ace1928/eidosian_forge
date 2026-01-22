import pyopencl as cl
import numpy as np
import functools
import os
import logging
from typing import Any, Dict, Tuple
from collections import deque
import pickle
import shutil
import OpenGL.GL as gl
def load_shader_from_file(self, filename: str='', shader_type: str='vertex') -> Any:
    """
        Loads a shader from a file, using a default shader if the specified file is not found, and compiles it using OpenGL.

        Parameters:
            filename (str): The filename of the shader file.
            shader_type (str): The type of shader to load.

        Returns:
            Any: The compiled OpenGL shader program.
        """
    if not filename:
        filename = self.default_shaders.get(shader_type, '')
        if not filename:
            error_message = f'No default shader file for type: {shader_type}'
            logging.error(error_message)
            raise ValueError(error_message)
    shader_path = os.path.join(self.default_shader_directory, filename)
    try:
        with open(shader_path, 'r') as file:
            shader_code = file.read()
        return self.compile_shader(shader_code, shader_type)
    except FileNotFoundError:
        logging.error(f'Shader file not found: {shader_path}')
        raise
    except Exception as e:
        logging.error(f'Error reading shader file {shader_path}: {str(e)}')
        raise