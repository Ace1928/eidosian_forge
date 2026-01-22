from pygsp.graphs import Graph, Grid2d, ImgPatches
Union of a patch graph with a 2D grid graph.

    Parameters
    ----------
    img : array
        Input image.
    aggregate: callable, optional
        Function to aggregate the weights ``Wp`` of the patch graph and the
        ``Wg`` of the grid graph. Default is ``lambda Wp, Wg: Wp + Wg``.
    kwargs : dict
        Parameters passed to :class:`ImgPatches`.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from skimage import data, img_as_float
    >>> img = img_as_float(data.camera()[::64, ::64])
    >>> G = graphs.Grid2dImgPatches(img)
    >>> fig, axes = plt.subplots(1, 2)
    >>> _ = axes[0].spy(G.W, markersize=2)
    >>> G.plot(ax=axes[1])

    