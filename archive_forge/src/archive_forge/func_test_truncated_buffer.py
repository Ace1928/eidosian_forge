from io import BytesIO
from pathlib import Path
import pytest
from matplotlib.testing.decorators import image_comparison
from matplotlib import cm, pyplot as plt
def test_truncated_buffer():
    b = BytesIO()
    plt.savefig(b)
    b.seek(0)
    b2 = BytesIO(b.read(20))
    b2.seek(0)
    with pytest.raises(Exception):
        plt.imread(b2)