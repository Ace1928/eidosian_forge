import os.path as op
from ..utils.filemanip import split_filename
from .base import (

    MeshFix v1.2-alpha - by Marco Attene, Mirko Windhoff, Axel Thielscher.

    .. seealso::

        http://jmeshlib.sourceforge.net
            Sourceforge page

        http://simnibs.de/installation/meshfixandgetfem
            Ubuntu installation instructions

    If MeshFix is used for research purposes, please cite the following paper:
    M. Attene - A lightweight approach to repairing digitized polygon meshes.
    The Visual Computer, 2010. (c) Springer.

    Accepted input formats are OFF, PLY and STL.
    Other formats (like .msh for gmsh) are supported only partially.

    Example
    -------

    >>> import nipype.interfaces.meshfix as mf
    >>> fix = mf.MeshFix()
    >>> fix.inputs.in_file1 = 'lh-pial.stl'
    >>> fix.inputs.in_file2 = 'rh-pial.stl'
    >>> fix.run()                                    # doctest: +SKIP
    >>> fix.cmdline
    'meshfix lh-pial.stl rh-pial.stl -o lh-pial_fixed.off'
    