from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def ll_fz_process_shade(shade, ctm, scissor, prepare, process, process_arg):
    """
    Low-level wrapper for `::fz_process_shade()`.
    Process a shade, using supplied callback functions. This
    decomposes the shading to a mesh (even ones that are not
    natively meshes, such as linear or radial shadings), and
    processes triangles from those meshes.

    shade: The shade to process.

    ctm: The transform to use

    prepare: Callback function to 'prepare' each vertex.
    This function is passed an array of floats, and populates
    a fz_vertex structure.

    process: This function is passed 3 pointers to vertex
    structures, and actually performs the processing (typically
    filling the area between the vertexes).

    process_arg: An opaque argument passed through from caller
    to callback functions.
    """
    return _mupdf.ll_fz_process_shade(shade, ctm, scissor, prepare, process, process_arg)