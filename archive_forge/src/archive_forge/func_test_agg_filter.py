import io
import numpy as np
from numpy.testing import assert_array_almost_equal
from PIL import Image, TiffTags
import pytest
from matplotlib import (
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from matplotlib.image import imread
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison
from matplotlib.transforms import IdentityTransform
@image_comparison(['agg_filter.png'], remove_text=True)
def test_agg_filter():

    def smooth1d(x, window_len):
        s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
        w = np.hanning(window_len)
        y = np.convolve(w / w.sum(), s, mode='same')
        return y[window_len - 1:-window_len + 1]

    def smooth2d(A, sigma=3):
        window_len = max(int(sigma), 3) * 2 + 1
        A = np.apply_along_axis(smooth1d, 0, A, window_len)
        A = np.apply_along_axis(smooth1d, 1, A, window_len)
        return A

    class BaseFilter:

        def get_pad(self, dpi):
            return 0

        def process_image(self, padded_src, dpi):
            raise NotImplementedError('Should be overridden by subclasses')

        def __call__(self, im, dpi):
            pad = self.get_pad(dpi)
            padded_src = np.pad(im, [(pad, pad), (pad, pad), (0, 0)], 'constant')
            tgt_image = self.process_image(padded_src, dpi)
            return (tgt_image, -pad, -pad)

    class OffsetFilter(BaseFilter):

        def __init__(self, offsets=(0, 0)):
            self.offsets = offsets

        def get_pad(self, dpi):
            return int(max(self.offsets) / 72 * dpi)

        def process_image(self, padded_src, dpi):
            ox, oy = self.offsets
            a1 = np.roll(padded_src, int(ox / 72 * dpi), axis=1)
            a2 = np.roll(a1, -int(oy / 72 * dpi), axis=0)
            return a2

    class GaussianFilter(BaseFilter):
        """Simple Gaussian filter."""

        def __init__(self, sigma, alpha=0.5, color=(0, 0, 0)):
            self.sigma = sigma
            self.alpha = alpha
            self.color = color

        def get_pad(self, dpi):
            return int(self.sigma * 3 / 72 * dpi)

        def process_image(self, padded_src, dpi):
            tgt_image = np.empty_like(padded_src)
            tgt_image[:, :, :3] = self.color
            tgt_image[:, :, 3] = smooth2d(padded_src[:, :, 3] * self.alpha, self.sigma / 72 * dpi)
            return tgt_image

    class DropShadowFilter(BaseFilter):

        def __init__(self, sigma, alpha=0.3, color=(0, 0, 0), offsets=(0, 0)):
            self.gauss_filter = GaussianFilter(sigma, alpha, color)
            self.offset_filter = OffsetFilter(offsets)

        def get_pad(self, dpi):
            return max(self.gauss_filter.get_pad(dpi), self.offset_filter.get_pad(dpi))

        def process_image(self, padded_src, dpi):
            t1 = self.gauss_filter.process_image(padded_src, dpi)
            t2 = self.offset_filter.process_image(t1, dpi)
            return t2
    fig, ax = plt.subplots()
    line1, = ax.plot([0.1, 0.5, 0.9], [0.1, 0.9, 0.5], 'bo-', mec='b', mfc='w', lw=5, mew=3, ms=10, label='Line 1')
    line2, = ax.plot([0.1, 0.5, 0.9], [0.5, 0.2, 0.7], 'ro-', mec='r', mfc='w', lw=5, mew=3, ms=10, label='Line 1')
    gauss = DropShadowFilter(4)
    for line in [line1, line2]:
        xx = line.get_xdata()
        yy = line.get_ydata()
        shadow, = ax.plot(xx, yy)
        shadow.update_from(line)
        transform = mtransforms.offset_copy(line.get_transform(), ax.figure, x=4.0, y=-6.0, units='points')
        shadow.set_transform(transform)
        shadow.set_zorder(line.get_zorder() - 0.5)
        shadow.set_agg_filter(gauss)
        shadow.set_rasterized(True)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)