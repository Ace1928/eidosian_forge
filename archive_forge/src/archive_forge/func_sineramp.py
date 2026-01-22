import numpy as np
def sineramp(size=(256, 512), amp=12.5, wavelen=8, p=2):
    if len(size) == 1:
        rows = cols = size
    elif len(size) == 2:
        rows, cols = size
    else:
        raise ValueError('size must be of length 1 or 2')
    cycles = int(cols / wavelen)
    cols = cycles * wavelen
    fx = amp * np.array([np.sin(1 / wavelen * 2 * np.pi * c) for c in range(cols)])
    A = (np.arange(rows, 0, -1) / (rows - 1)) ** p
    im_0, im_1 = np.meshgrid(fx, A)
    im = im_0 * im_1
    ramp_0, ramp_1 = np.meshgrid(range(cols), range(rows))
    ramp = ramp_0 / cols
    im = im + ramp * (255 - 2 * amp)
    for r in range(rows):
        im[r, :] = im[r, :] / im[r, :].max()
    return im * 255