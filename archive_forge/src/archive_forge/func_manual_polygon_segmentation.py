from functools import reduce
import numpy as np
from ..draw import polygon
from .._shared.version_requirements import require
@require('matplotlib', '>=3.3')
def manual_polygon_segmentation(image, alpha=0.4, return_all=False):
    """Return a label image based on polygon selections made with the mouse.

    Parameters
    ----------
    image : (M, N[, 3]) array
        Grayscale or RGB image.

    alpha : float, optional
        Transparency value for polygons drawn over the image.

    return_all : bool, optional
        If True, an array containing each separate polygon drawn is returned.
        (The polygons may overlap.) If False (default), latter polygons
        "overwrite" earlier ones where they overlap.

    Returns
    -------
    labels : array of int, shape ([Q, ]M, N)
        The segmented regions. If mode is `'separate'`, the leading dimension
        of the array corresponds to the number of regions that the user drew.

    Notes
    -----
    Use left click to select the vertices of the polygon
    and right click to confirm the selection once all vertices are selected.

    Examples
    --------
    >>> from skimage import data, future, io
    >>> camera = data.camera()
    >>> mask = future.manual_polygon_segmentation(camera)  # doctest: +SKIP
    >>> io.imshow(mask)  # doctest: +SKIP
    >>> io.show()  # doctest: +SKIP
    """
    import matplotlib
    import matplotlib.pyplot as plt
    list_of_vertex_lists = []
    polygons_drawn = []
    temp_list = []
    preview_polygon_drawn = []
    if image.ndim not in (2, 3):
        raise ValueError('Only 2D grayscale or RGB images are supported.')
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.2)
    ax.imshow(image, cmap='gray')
    ax.set_axis_off()

    def _undo(*args, **kwargs):
        if list_of_vertex_lists:
            list_of_vertex_lists.pop()
            last_poly = polygons_drawn.pop()
            last_poly.remove()
            fig.canvas.draw_idle()
    undo_pos = fig.add_axes([0.85, 0.05, 0.075, 0.075])
    undo_button = matplotlib.widgets.Button(undo_pos, '⟲')
    undo_button.on_clicked(_undo)

    def _extend_polygon(event):
        if event.inaxes is None or event.inaxes is undo_pos:
            return
        if ax.get_navigate_mode():
            return
        if event.button == LEFT_CLICK:
            temp_list.append([event.xdata, event.ydata])
            if preview_polygon_drawn:
                poly = preview_polygon_drawn.pop()
                poly.remove()
            polygon = _draw_polygon(ax, temp_list, alpha=alpha / 1.4)
            preview_polygon_drawn.append(polygon)
        elif event.button == RIGHT_CLICK:
            if not temp_list:
                return
            list_of_vertex_lists.append(temp_list[:])
            polygon_object = _draw_polygon(ax, temp_list, alpha=alpha)
            polygons_drawn.append(polygon_object)
            preview_poly = preview_polygon_drawn.pop()
            preview_poly.remove()
            del temp_list[:]
            plt.draw()
    fig.canvas.mpl_connect('button_press_event', _extend_polygon)
    plt.show(block=True)
    labels = (_mask_from_vertices(vertices, image.shape[:2], i) for i, vertices in enumerate(list_of_vertex_lists, start=1))
    if return_all:
        return np.stack(labels)
    else:
        return reduce(np.maximum, labels, np.broadcast_to(0, image.shape[:2]))