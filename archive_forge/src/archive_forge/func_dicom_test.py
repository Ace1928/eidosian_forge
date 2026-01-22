from __future__ import annotations
import warnings
from typing import Callable
from .deprecated import deprecate_with_version
from .optpkg import optional_package
@deprecate_with_version('dicom_test has been moved to nibabel.nicom.tests', since='3.1', until='5.0')
def dicom_test(func):
    from .nicom.tests import dicom_test
    return dicom_test(func)