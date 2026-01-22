from pathlib import Path
import numpy as np
import pytest
from ase.utils.xrdebye import XrDebye, wavelengths
from ase.cluster.cubic import FaceCenteredCubic
def test_saxs_and_files(testdir, figure, xrd):
    expected = np.array([372650934.006398, 280252013.563702, 488123.103628])
    obtained = xrd.calc_pattern(x=np.array([0.021, 0.09, 0.53]), mode='SAXS')
    assert np.allclose(obtained, expected, rtol=tolerance)
    xrd.write_pattern('tmp.txt')
    assert Path('tmp.txt').exists()
    ax = figure.add_subplot(111)
    xrd.plot_pattern(ax=ax, filename='pattern.png')
    assert Path('pattern.png').exists()