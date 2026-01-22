from ...testing import utils
from ..confounds import TSNR
from .. import misc
import pytest
import numpy.testing as npt
from unittest import mock
import nibabel as nb
import numpy as np
import os
test that usage of misc.TSNR trips a warning to use
        confounds.TSNR instead