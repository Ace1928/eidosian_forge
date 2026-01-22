import pytest
import platform
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
import matplotlib.patches as mpatches
@image_comparison(['arrow_styles.png'], style='mpl20', remove_text=True, tol=0 if platform.machine() == 'x86_64' else 0.005)
def test_arrow_styles():
    styles = mpatches.ArrowStyle.get_styles()
    n = len(styles)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, n)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    for i, stylename in enumerate(sorted(styles)):
        patch = mpatches.FancyArrowPatch((0.1 + i % 2 * 0.05, i), (0.45 + i % 2 * 0.05, i), arrowstyle=stylename, mutation_scale=25)
        ax.add_patch(patch)
    for i, stylename in enumerate([']-[', ']-', '-[', '|-|']):
        style = stylename
        if stylename[0] != '-':
            style += ',angleA=ANGLE'
        if stylename[-1] != '-':
            style += ',angleB=ANGLE'
        for j, angle in enumerate([-30, 60]):
            arrowstyle = style.replace('ANGLE', str(angle))
            patch = mpatches.FancyArrowPatch((0.55, 2 * i + j), (0.9, 2 * i + j), arrowstyle=arrowstyle, mutation_scale=25)
            ax.add_patch(patch)