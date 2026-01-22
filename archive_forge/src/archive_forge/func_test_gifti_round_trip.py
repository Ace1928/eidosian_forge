import itertools
import sys
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nibabel.tmpdirs import InTemporaryDirectory
from ... import load
from ...fileholders import FileHolder
from ...nifti1 import data_type_codes
from ...testing import get_test_data
from .. import (
from .test_parse_gifti_fast import (
def test_gifti_round_trip():
    test_data = b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE GIFTI SYSTEM "http://www.nitrc.org/frs/download.php/1594/gifti.dtd">\n<GIFTI\nxmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\nxsi:noNamespaceSchemaLocation="http://www.nitrc.org/frs/download.php/1303/GIFTI_Caret.xsd"\nVersion="1.0"\nNumberOfDataArrays="2">\n<MetaData>\n<MD>\n<Name><![CDATA[date]]></Name>\n<Value><![CDATA[Thu Nov 15 09:05:22 2007]]></Value>\n</MD>\n</MetaData>\n<LabelTable/>\n<DataArray Intent="NIFTI_INTENT_POINTSET"\nDataType="NIFTI_TYPE_FLOAT32"\nArrayIndexingOrder="RowMajorOrder"\nDimensionality="2"\nDim0="4"\nDim1="3"\nEncoding="ASCII"\nEndian="LittleEndian"\nExternalFileName=""\nExternalFileOffset="">\n<CoordinateSystemTransformMatrix>\n<DataSpace><![CDATA[NIFTI_XFORM_TALAIRACH]]></DataSpace>\n<TransformedSpace><![CDATA[NIFTI_XFORM_TALAIRACH]]></TransformedSpace>\n<MatrixData>\n1.000000 0.000000 0.000000 0.000000\n0.000000 1.000000 0.000000 0.000000\n0.000000 0.000000 1.000000 0.000000\n0.000000 0.000000 0.000000 1.000000\n</MatrixData>\n</CoordinateSystemTransformMatrix>\n<Data>\n10.5 0 0\n0 20.5 0\n0 0 30.5\n0 0 0\n</Data>\n</DataArray>\n<DataArray Intent="NIFTI_INTENT_TRIANGLE"\nDataType="NIFTI_TYPE_INT32"\nArrayIndexingOrder="RowMajorOrder"\nDimensionality="2"\nDim0="4"\nDim1="3"\nEncoding="ASCII"\nEndian="LittleEndian"\nExternalFileName="" ExternalFileOffset="">\n<Data>\n0 1 2\n1 2 3\n0 1 3\n0 2 3\n</Data>\n</DataArray>\n</GIFTI>'
    exp_verts = np.zeros((4, 3))
    exp_verts[0, 0] = 10.5
    exp_verts[1, 1] = 20.5
    exp_verts[2, 2] = 30.5
    exp_faces = np.asarray([[0, 1, 2], [1, 2, 3], [0, 1, 3], [0, 2, 3]], dtype=np.int32)

    def _check_gifti(gio):
        vertices = gio.get_arrays_from_intent('NIFTI_INTENT_POINTSET')[0].data
        faces = gio.get_arrays_from_intent('NIFTI_INTENT_TRIANGLE')[0].data
        assert_array_equal(vertices, exp_verts)
        assert_array_equal(faces, exp_faces)
    bio = BytesIO()
    fmap = dict(image=FileHolder(fileobj=bio))
    bio.write(test_data)
    bio.seek(0)
    gio = GiftiImage.from_file_map(fmap)
    _check_gifti(gio)
    bio.seek(0)
    gio.to_file_map(fmap)
    bio.seek(0)
    gio2 = GiftiImage.from_file_map(fmap)
    _check_gifti(gio2)