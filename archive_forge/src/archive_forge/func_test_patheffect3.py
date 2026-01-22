import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.path import Path
import matplotlib.patches as patches
@image_comparison(['patheffect3'])
def test_patheffect3():
    p1, = plt.plot([1, 3, 5, 4, 3], 'o-b', lw=4)
    p1.set_path_effects([path_effects.SimpleLineShadow(), path_effects.Normal()])
    plt.title('testing$^{123}$', path_effects=[path_effects.withStroke(linewidth=1, foreground='r')])
    leg = plt.legend([p1], ['Line 1$^2$'], fancybox=True, loc='upper left')
    leg.legendPatch.set_path_effects([path_effects.withSimplePatchShadow()])
    text = plt.text(2, 3, 'Drop test', color='white', bbox={'boxstyle': 'circle,pad=0.1', 'color': 'red'})
    pe = [path_effects.Stroke(linewidth=3.75, foreground='k'), path_effects.withSimplePatchShadow((6, -3), shadow_rgbFace='blue')]
    text.set_path_effects(pe)
    text.get_bbox_patch().set_path_effects(pe)
    pe = [path_effects.PathPatchEffect(offset=(4, -4), hatch='xxxx', facecolor='gray'), path_effects.PathPatchEffect(edgecolor='white', facecolor='black', lw=1.1)]
    t = plt.gcf().text(0.02, 0.1, 'Hatch shadow', fontsize=75, weight=1000, va='center')
    t.set_path_effects(pe)