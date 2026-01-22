import pyopencl as cl  # https://documen.tician.de/pyopencl/ - Used for managing and executing OpenCL commands on GPUs.
import OpenGL.GL as gl  # https://pyopengl.sourceforge.io/documentation/ - Used for executing OpenGL commands for rendering graphics.
import json  # https://docs.python.org/3/library/json.html - Used for parsing and outputting JSON formatted data.
import numpy as np  # https://numpy.org/doc/ - Used for numerical operations on arrays and matrices.
import functools  # https://docs.python.org/3/library/functools.html - Provides higher-order functions and operations on callable objects.
import logging  # https://docs.python.org/3/library/logging.html - Used for logging events and messages during execution.
from pyopencl import (
import hashlib  # https://docs.python.org/3/library/hashlib.html - Used for hashing algorithms.
import pickle  # https://docs.python.org/3/library/pickle.html - Used for serializing and deserializing Python objects.
from typing import (
from functools import (
@lru_cache(maxsize=128)
def create_program(self, source_code: str) -> cl.Program:
    """
        Compiles OpenCL source code into a program using the GPU context provided by the GPUManager.

        Parameters:
            source_code (str): The OpenCL source code as a string.

        Returns:
            cl.Program: The compiled OpenCL program.

        Raises:
            cl.ProgramBuildError: If there is an error during the building of the OpenCL program.
        """
    try:
        program = cl.Program(self.gpu_manager.context, source_code).build()
        logging.info('OpenCL program created and built from source.')
        return program
    except cl.ProgramBuildError as e:
        logging.error(f'Failed to build OpenCL program: {e}')
        raise