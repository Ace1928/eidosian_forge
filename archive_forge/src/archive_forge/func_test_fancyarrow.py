import pytest
import platform
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.patches as mpatches
@image_comparison(['fancyarrow_test_image'])
def test_fancyarrow():
    r = [0.4, 0.3, 0.2, 0.1, 0]
    t = ['fancy', 'simple', mpatches.ArrowStyle.Fancy()]
    fig, axs = plt.subplots(len(t), len(r), squeeze=False, figsize=(8, 4.5), subplot_kw=dict(aspect=1))
    for i_r, r1 in enumerate(r):
        for i_t, t1 in enumerate(t):
            ax = axs[i_t, i_r]
            draw_arrow(ax, t1, r1)
            ax.tick_params(labelleft=False, labelbottom=False)