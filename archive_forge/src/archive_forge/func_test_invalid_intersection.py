import pytest
import platform
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.patches as mpatches
def test_invalid_intersection():
    conn_style_1 = mpatches.ConnectionStyle.Angle3(angleA=20, angleB=200)
    p1 = mpatches.FancyArrowPatch((0.2, 0.2), (0.5, 0.5), connectionstyle=conn_style_1)
    with pytest.raises(ValueError):
        plt.gca().add_patch(p1)
    conn_style_2 = mpatches.ConnectionStyle.Angle3(angleA=20, angleB=199.9)
    p2 = mpatches.FancyArrowPatch((0.2, 0.2), (0.5, 0.5), connectionstyle=conn_style_2)
    plt.gca().add_patch(p2)