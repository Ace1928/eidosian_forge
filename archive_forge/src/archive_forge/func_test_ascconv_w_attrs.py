from collections import OrderedDict
from os.path import dirname
from os.path import join as pjoin
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from .. import ascconv
def test_ascconv_w_attrs():
    in_str = '### ASCCONV BEGIN object=MrProtDataImpl@MrProtocolData version=41340006 converter=%MEASCONST%/ConverterList/Prot_Converter.txt ###\ntest = "hello"\n### ASCCONV END ###'
    ascconv_dict, attrs = ascconv.parse_ascconv(in_str, '""')
    assert attrs['object'] == 'MrProtDataImpl@MrProtocolData'
    assert attrs['version'] == '41340006'
    assert attrs['converter'] == '%MEASCONST%/ConverterList/Prot_Converter.txt'
    assert ascconv_dict['test'] == 'hello'