from colorsys import hsv_to_rgb, hls_to_rgb
from .libmp import NoConvergence
from .libmp.backend import xrange
def default_color_function(ctx, z):
    if ctx.isinf(z):
        return (1.0, 1.0, 1.0)
    if ctx.isnan(z):
        return (0.5, 0.5, 0.5)
    pi = 3.1415926535898
    a = (float(ctx.arg(z)) + ctx.pi) / (2 * ctx.pi)
    a = (a + 0.5) % 1.0
    b = 1.0 - float(1 / (1.0 + abs(z) ** 0.3))
    return hls_to_rgb(a, b, 0.8)