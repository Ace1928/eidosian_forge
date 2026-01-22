import numpy as np
from skimage.graph._graph import pixel_graph, central_pixel
def edge_func(values_src, values_dst, distances):
    return np.abs(values_src - values_dst) + distances