from plotly import exceptions, optional_imports
import plotly.colors as clrs
from plotly.graph_objs import graph_objs
def trisurf(x, y, z, simplices, show_colorbar, edges_color, scale, colormap=None, color_func=None, plot_edges=False, x_edge=None, y_edge=None, z_edge=None, facecolor=None):
    """
    Refer to FigureFactory.create_trisurf() for docstring
    """
    if not np:
        raise ImportError('FigureFactory._trisurf() requires numpy imported.')
    points3D = np.vstack((x, y, z)).T
    simplices = np.atleast_2d(simplices)
    tri_vertices = points3D[simplices]
    if color_func is None:
        mean_dists = tri_vertices[:, :, 2].mean(-1)
    elif isinstance(color_func, (list, np.ndarray)):
        if len(color_func) != len(simplices):
            raise ValueError('If color_func is a list/array, it must be the same length as simplices.')
        for index in range(len(color_func)):
            if isinstance(color_func[index], str):
                if '#' in color_func[index]:
                    foo = clrs.hex_to_rgb(color_func[index])
                    color_func[index] = clrs.label_rgb(foo)
            if isinstance(color_func[index], tuple):
                foo = clrs.convert_to_RGB_255(color_func[index])
                color_func[index] = clrs.label_rgb(foo)
        mean_dists = np.asarray(color_func)
    else:
        mean_dists = []
        for triangle in tri_vertices:
            dists = []
            for vertex in triangle:
                dist = color_func(vertex[0], vertex[1], vertex[2])
                dists.append(dist)
            mean_dists.append(np.mean(dists))
        mean_dists = np.asarray(mean_dists)
    if isinstance(mean_dists[0], str):
        facecolor = mean_dists
    else:
        min_mean_dists = np.min(mean_dists)
        max_mean_dists = np.max(mean_dists)
        if facecolor is None:
            facecolor = []
        for index in range(len(mean_dists)):
            color = map_face2color(mean_dists[index], colormap, scale, min_mean_dists, max_mean_dists)
            facecolor.append(color)
    facecolor = np.asarray(facecolor)
    ii, jj, kk = simplices.T
    triangles = graph_objs.Mesh3d(x=x, y=y, z=z, facecolor=facecolor, i=ii, j=jj, k=kk, name='')
    mean_dists_are_numbers = not isinstance(mean_dists[0], str)
    if mean_dists_are_numbers and show_colorbar is True:
        colorscale = clrs.make_colorscale(colormap, scale)
        colorscale = clrs.convert_colorscale_to_rgb(colorscale)
        colorbar = graph_objs.Scatter3d(x=x[:1], y=y[:1], z=z[:1], mode='markers', marker=dict(size=0.1, color=[min_mean_dists, max_mean_dists], colorscale=colorscale, showscale=True), hoverinfo='none', showlegend=False)
    if plot_edges is False:
        if mean_dists_are_numbers and show_colorbar is True:
            return [triangles, colorbar]
        else:
            return [triangles]
    is_none = [ii is None for ii in [x_edge, y_edge, z_edge]]
    if any(is_none):
        if not all(is_none):
            raise ValueError('If any (x_edge, y_edge, z_edge) is None, all must be None')
        else:
            x_edge = []
            y_edge = []
            z_edge = []
    ixs_triangles = [0, 1, 2, 0]
    pull_edges = tri_vertices[:, ixs_triangles, :]
    x_edge_pull = np.hstack([pull_edges[:, :, 0], np.tile(None, [pull_edges.shape[0], 1])])
    y_edge_pull = np.hstack([pull_edges[:, :, 1], np.tile(None, [pull_edges.shape[0], 1])])
    z_edge_pull = np.hstack([pull_edges[:, :, 2], np.tile(None, [pull_edges.shape[0], 1])])
    x_edge = np.hstack([x_edge, x_edge_pull.reshape([1, -1])[0]])
    y_edge = np.hstack([y_edge, y_edge_pull.reshape([1, -1])[0]])
    z_edge = np.hstack([z_edge, z_edge_pull.reshape([1, -1])[0]])
    if not len(x_edge) == len(y_edge) == len(z_edge):
        raise exceptions.PlotlyError('The lengths of x_edge, y_edge and z_edge are not the same.')
    lines = graph_objs.Scatter3d(x=x_edge, y=y_edge, z=z_edge, mode='lines', line=graph_objs.scatter3d.Line(color=edges_color, width=1.5), showlegend=False)
    if mean_dists_are_numbers and show_colorbar is True:
        return [triangles, lines, colorbar]
    else:
        return [triangles, lines]