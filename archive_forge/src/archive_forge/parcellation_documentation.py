import os
import os.path as op
import shutil
import numpy as np
import nibabel as nb
import networkx as nx
from ... import logging
from ..base import (
from .base import have_cmp
Subdivides segmented ROI file into smaller subregions

    This interface implements the same procedure as in the ConnectomeMapper's
    parcellation stage (cmp/stages/parcellation/maskcreation.py) for a single
    parcellation scheme (e.g. 'scale500').

    Example
    -------

    >>> import nipype.interfaces.cmtk as cmtk
    >>> parcellate = cmtk.Parcellate()
    >>> parcellate.inputs.freesurfer_dir = '.'
    >>> parcellate.inputs.subjects_dir = '.'
    >>> parcellate.inputs.subject_id = 'subj1'
    >>> parcellate.inputs.dilation = True
    >>> parcellate.inputs.parcellation_name = 'scale500'
    >>> parcellate.run()                 # doctest: +SKIP
    