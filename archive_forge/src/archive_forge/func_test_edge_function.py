import numpy as np
from skimage.graph._graph import pixel_graph, central_pixel
def test_edge_function():

    def edge_func(values_src, values_dst, distances):
        return np.abs(values_src - values_dst) + distances
    g, n = pixel_graph(image, mask=mask, connectivity=2, edge_function=edge_func)
    s2 = np.sqrt(2)
    np.testing.assert_allclose(g[0, 1], np.abs(image[0, 0] - image[1, 1]) + s2)
    np.testing.assert_allclose(g[1, 2], np.abs(image[1, 1] - image[1, 2]) + 1)
    np.testing.assert_array_equal(n, [0, 4, 5, 7])