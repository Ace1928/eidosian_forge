from collections import OrderedDict
from os.path import dirname
from os.path import join as pjoin
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import ascconv
def test_ascconv_parse():
    with open(ASCCONV_INPUT) as fobj:
        contents = fobj.read()
    ascconv_dict, attrs = ascconv.parse_ascconv(contents, str_delim='""')
    assert attrs == OrderedDict()
    assert len(ascconv_dict) == 72
    assert ascconv_dict['tProtocolName'] == 'CBU+AF8-DTI+AF8-64D+AF8-1A'
    assert ascconv_dict['ucScanRegionPosValid'] == 1
    assert_array_almost_equal(ascconv_dict['sProtConsistencyInfo']['flNominalB0'], 2.89362)
    assert ascconv_dict['sProtConsistencyInfo']['flGMax'] == 26
    assert list(ascconv_dict['sSliceArray'].keys()) == ['asSlice', 'anAsc', 'anPos', 'lSize', 'lConc', 'ucMode', 'sTSat']
    slice_arr = ascconv_dict['sSliceArray']
    as_slice = slice_arr['asSlice']
    assert_array_equal([e['dPhaseFOV'] for e in as_slice], 230)
    assert_array_equal([e['dReadoutFOV'] for e in as_slice], 230)
    assert_array_equal([e['dThickness'] for e in as_slice], 2.5)
    assert slice_arr['anAsc'] == [None] + list(range(1, 48))
    assert slice_arr['anPos'] == [None] + list(range(1, 48))
    assert len(ascconv_dict['asCoilSelectMeas']) == 1
    as_list = ascconv_dict['asCoilSelectMeas'][0]['asList']
    assert len(as_list) == 12
    for i, el in enumerate(as_list):
        assert list(el.keys()) == ['sCoilElementID', 'lElementSelected', 'lRxChannelConnected']
        assert el['lElementSelected'] == 1
        assert el['lRxChannelConnected'] == i + 1
    assert_array_almost_equal(as_slice[0]['sPosition']['dCor'], -20.03015269)