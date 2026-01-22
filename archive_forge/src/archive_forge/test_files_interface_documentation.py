from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from .. import MGHImage, Nifti1Image, Nifti1Pair, all_image_classes
from ..fileholders import FileHolderError
from ..spatialimages import SpatialImage
Testing filesets - a draft
