import os
from .. import logging
def vtk_output(obj):
    """Configure the input data for vtk pipeline object obj."""
    if vtk_old():
        return obj.output
    return obj.get_output()