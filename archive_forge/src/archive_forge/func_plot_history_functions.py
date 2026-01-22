import itertools
import functools
import importlib.util
@default_to_neutral_style
def plot_history_functions(self, *, fn=None, log=None, colors=None, kind='scatter', scatter_size=5, scatter_marker='s', lines_width=5, image_alpha_pow=2 / 3, image_aspect=4, legend=True, legend_ncol=None, legend_bbox_to_anchor=None, legend_loc=None, rasterize=4096, rasterize_dpi=300, ax=None, figsize=(8, 2), show_and_close=True):
    """Plot the functions used throughout this computation, color coded, as
    either a scatter plot or an image, showing the size of the that individual
    intermediate as well.
    """
    import numpy as np
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    if fn is not None:
        ylabel = 'custom'
    else:
        ylabel = 'node size'

        def fn(node):
            return node.size
    if log:
        ylabel = f'$\\log_{{{log}}}[{ylabel}]$'
        orig_fn = fn

        def fn(node):
            return np.log2(orig_fn(node)) / np.log2(log)
    colors = get_default_colors_dict(colors)
    xs = []
    ys = []
    cs = []
    ymax = 0
    for i, node in enumerate(self.ascend()):
        xs.append(i)
        y = fn(node)
        ymax = max(ymax, y)
        ys.append(y)
        try:
            c = colors[node.fn_name]
        except KeyError:
            c = colors[node.fn_name] = hash_to_color(node.fn_name)
        cs.append(c)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.set_dpi(rasterize_dpi)
        ax.set_ylabel(ylabel)
    else:
        fig = None
    if isinstance(rasterize, (float, int)):
        rasterize = len(xs) > rasterize
    if rasterize:
        ax.set_rasterization_zorder(0)
    if kind == 'scatter':
        ax.scatter(xs, ys, c=cs, s=scatter_size, marker=scatter_marker, rasterized=rasterize)
    elif kind == 'lines':
        lns = [((x, 0.0), (x, y)) for x, y in zip(xs, ys)]
        ax.add_collection(mpl.collections.LineCollection(lns, colors=cs, zorder=-1, lw=lines_width))
        ax.set_xlim(-0.5, len(lns) + 0.5)
        ax.set_ylim(0, 1.05 * ymax)
    elif kind == 'image':
        ax.axis('off')
        ys = np.array(ys)
        ys = (ys / ys.max()).reshape(-1, 1) ** image_alpha_pow
        N = len(cs)
        da = round((N / image_aspect) ** 0.5)
        db = N // da
        while da * db < N:
            db += 1
        Ns = da * db
        img = np.concatenate([cs, ys], axis=1)
        img = np.concatenate([img, np.tile(0.0, (Ns - N, 4))], axis=0)
        img = img.reshape(da, db, 4)
        ax.imshow(img, zorder=-1)
    if legend:
        legend_items = [mpl.patches.Patch(facecolor=c, label=fn_name) for fn_name, c in colors.items()]
        if legend_ncol is None:
            legend_ncol = max(1, round(len(legend_items) / 6))
        if legend_bbox_to_anchor is None:
            legend_bbox_to_anchor = (1.0, 1.0)
        if legend_loc is None:
            legend_loc = 'upper left'
        ax.legend(handles=legend_items, ncol=legend_ncol, bbox_to_anchor=legend_bbox_to_anchor, loc=legend_loc)
    if fig is not None and show_and_close:
        plt.show()
        plt.close(fig)
    return (fig, ax)