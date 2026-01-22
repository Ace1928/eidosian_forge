import numpy as np
import PIL.Image
import pythreejs
import scipy.interpolate
import ipyvolume as ipv
from ipyvolume.datasets import UrlCached
def radial_sprite(shape, color):
    color = np.array(color)
    ara = np.zeros(shape[:2] + (4,), dtype=np.uint8)
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    x, y = np.meshgrid(x, y)
    s = 0.5
    radius = np.sqrt(x ** 2 + y ** 2)
    amplitude = np.maximum(0, np.exp(-radius ** 2 / s ** 2)).T
    ara[..., 3] = amplitude * 255
    ara[..., :3] = color * amplitude.reshape(shape + (1,))
    im = PIL.Image.fromarray(ara, 'RGBA')
    return im