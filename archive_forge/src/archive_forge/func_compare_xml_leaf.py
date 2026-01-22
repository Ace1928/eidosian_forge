import collections
from xml.etree import ElementTree
import numpy as np
import pytest
from nibabel import cifti2 as ci
from nibabel.cifti2.cifti2 import Cifti2HeaderError, _float_01, _value_if_klass
from nibabel.nifti2 import Nifti2Header
from nibabel.tests.test_dataobj_images import TestDataobjAPI as _TDA
from nibabel.tests.test_image_api import DtypeOverrideMixin, SerializeMixin
def compare_xml_leaf(str1, str2):
    x1 = ElementTree.fromstring(str1)
    x2 = ElementTree.fromstring(str2)
    if len(x1) > 0 or len(x2) > 0:
        raise ValueError
    test = x1.tag == x2.tag and x1.attrib == x2.attrib and (x1.text == x2.text)
    print((x1.tag, x1.attrib, x1.text))
    print((x2.tag, x2.attrib, x2.text))
    return test