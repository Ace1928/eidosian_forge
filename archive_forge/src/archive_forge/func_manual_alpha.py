import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison
def manual_alpha(im, dpi):
    im[:, :, 3] *= 0.6
    print('CALLED')
    return (im, 0, 0)