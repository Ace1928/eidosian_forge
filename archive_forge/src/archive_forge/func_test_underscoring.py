import collections
from xml.etree import ElementTree
import numpy as np
import pytest
from nibabel import cifti2 as ci
from nibabel.cifti2.cifti2 import Cifti2HeaderError, _float_01, _value_if_klass
from nibabel.nifti2 import Nifti2Header
from nibabel.tests.test_dataobj_images import TestDataobjAPI as _TDA
from nibabel.tests.test_image_api import DtypeOverrideMixin, SerializeMixin
def test_underscoring():
    pairs = (('Product', 'product'), ('SpecialGuest', 'special_guest'), ('ApplicationController', 'application_controller'), ('Area51Controller', 'area51_controller'), ('HTMLTidy', 'html_tidy'), ('HTMLTidyGenerator', 'html_tidy_generator'), ('FreeBSD', 'free_bsd'), ('HTML', 'html'))
    for camel, underscored in pairs:
        assert ci.cifti2._underscore(camel) == underscored