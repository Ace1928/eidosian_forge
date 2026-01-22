import warnings
import pytest
from nibabel import pkg_info
from nibabel.deprecated import (
from nibabel.tests.test_deprecator import TestDeprecatorFunc as _TestDF
Test deprecations against nibabel version