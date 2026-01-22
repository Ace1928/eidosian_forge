import os
from .. import logging
def no_vtk():
    """Checks if VTK is installed and the python wrapper is functional"""
    global _vtk_version
    return _vtk_version is None