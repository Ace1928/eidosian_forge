import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.path import Path
import matplotlib.patches as patches
@image_comparison(['collection'], tol=0.03, style='mpl20')
def test_collection():
    x, y = np.meshgrid(np.linspace(0, 10, 150), np.linspace(-5, 5, 100))
    data = np.sin(x) + np.cos(y)
    cs = plt.contour(data)
    cs.set(path_effects=[path_effects.PathPatchEffect(edgecolor='black', facecolor='none', linewidth=12), path_effects.Stroke(linewidth=5)])
    for text in plt.clabel(cs, colors='white'):
        text.set_path_effects([path_effects.withStroke(foreground='k', linewidth=3)])
        text.set_bbox({'boxstyle': 'sawtooth', 'facecolor': 'none', 'edgecolor': 'blue'})