from pathlib import Path
import shutil
import pytest
from pytest import approx
from matplotlib.testing.compare import compare_images
from matplotlib.testing.decorators import _image_directories

    Compare two images, expecting a particular RMS error.

    im1 and im2 are filenames relative to the baseline_dir directory.

    tol is the tolerance to pass to compare_images.

    expect_rms is the expected RMS value, or None. If None, the test will
    succeed if compare_images succeeds. Otherwise, the test will succeed if
    compare_images fails and returns an RMS error almost equal to this value.
    