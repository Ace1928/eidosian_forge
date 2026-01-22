import math
import numpy as np
import scipy.ndimage as ndi
from .._shared.utils import check_nD, _supported_float_type
from ..feature.util import DescriptorExtractor, FeatureDetector
from .._shared.filters import gaussian
from ..transform import rescale
from ..util import img_as_float
from ._sift import _local_max, _ori_distances, _update_histogram
def detect_and_extract(self, image):
    """Detect the keypoints and extract their descriptors.

        Parameters
        ----------
        image : 2D array
            Input image.

        """
    image = self._preprocess(image)
    gaussian_scalespace = self._create_scalespace(image)
    dog_scalespace = [np.diff(layer, axis=2) for layer in gaussian_scalespace]
    positions, scales, sigmas, octaves = self._find_localize_evaluate(dog_scalespace, image.shape)
    gradient_space = self._compute_orientation(positions, scales, sigmas, octaves, gaussian_scalespace)
    self._compute_descriptor(gradient_space)
    self.keypoints = self.positions.round().astype(int)