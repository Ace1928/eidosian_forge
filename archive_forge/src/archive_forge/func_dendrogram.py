import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def dendrogram(Z, p=30, truncate_mode=None, color_threshold=None, get_leaves=True, orientation='top', labels=None, count_sort=False, distance_sort=False, show_leaf_counts=True, no_plot=False, no_labels=False, leaf_font_size=None, leaf_rotation=None, leaf_label_func=None, show_contracted=False, link_color_func=None, ax=None, above_threshold_color='C0'):
    """
    Plot the hierarchical clustering as a dendrogram.

    The dendrogram illustrates how each cluster is
    composed by drawing a U-shaped link between a non-singleton
    cluster and its children. The top of the U-link indicates a
    cluster merge. The two legs of the U-link indicate which clusters
    were merged. The length of the two legs of the U-link represents
    the distance between the child clusters. It is also the
    cophenetic distance between original observations in the two
    children clusters.

    Parameters
    ----------
    Z : ndarray
        The linkage matrix encoding the hierarchical clustering to
        render as a dendrogram. See the ``linkage`` function for more
        information on the format of ``Z``.
    p : int, optional
        The ``p`` parameter for ``truncate_mode``.
    truncate_mode : str, optional
        The dendrogram can be hard to read when the original
        observation matrix from which the linkage is derived is
        large. Truncation is used to condense the dendrogram. There
        are several modes:

        ``None``
          No truncation is performed (default).
          Note: ``'none'`` is an alias for ``None`` that's kept for
          backward compatibility.

        ``'lastp'``
          The last ``p`` non-singleton clusters formed in the linkage are the
          only non-leaf nodes in the linkage; they correspond to rows
          ``Z[n-p-2:end]`` in ``Z``. All other non-singleton clusters are
          contracted into leaf nodes.

        ``'level'``
          No more than ``p`` levels of the dendrogram tree are displayed.
          A "level" includes all nodes with ``p`` merges from the final merge.

          Note: ``'mtica'`` is an alias for ``'level'`` that's kept for
          backward compatibility.

    color_threshold : double, optional
        For brevity, let :math:`t` be the ``color_threshold``.
        Colors all the descendent links below a cluster node
        :math:`k` the same color if :math:`k` is the first node below
        the cut threshold :math:`t`. All links connecting nodes with
        distances greater than or equal to the threshold are colored
        with de default matplotlib color ``'C0'``. If :math:`t` is less
        than or equal to zero, all nodes are colored ``'C0'``.
        If ``color_threshold`` is None or 'default',
        corresponding with MATLAB(TM) behavior, the threshold is set to
        ``0.7*max(Z[:,2])``.

    get_leaves : bool, optional
        Includes a list ``R['leaves']=H`` in the result
        dictionary. For each :math:`i`, ``H[i] == j``, cluster node
        ``j`` appears in position ``i`` in the left-to-right traversal
        of the leaves, where :math:`j < 2n-1` and :math:`i < n`.
    orientation : str, optional
        The direction to plot the dendrogram, which can be any
        of the following strings:

        ``'top'``
          Plots the root at the top, and plot descendent links going downwards.
          (default).

        ``'bottom'``
          Plots the root at the bottom, and plot descendent links going
          upwards.

        ``'left'``
          Plots the root at the left, and plot descendent links going right.

        ``'right'``
          Plots the root at the right, and plot descendent links going left.

    labels : ndarray, optional
        By default, ``labels`` is None so the index of the original observation
        is used to label the leaf nodes.  Otherwise, this is an :math:`n`-sized
        sequence, with ``n == Z.shape[0] + 1``. The ``labels[i]`` value is the
        text to put under the :math:`i` th leaf node only if it corresponds to
        an original observation and not a non-singleton cluster.
    count_sort : str or bool, optional
        For each node n, the order (visually, from left-to-right) n's
        two descendent links are plotted is determined by this
        parameter, which can be any of the following values:

        ``False``
          Nothing is done.

        ``'ascending'`` or ``True``
          The child with the minimum number of original objects in its cluster
          is plotted first.

        ``'descending'``
          The child with the maximum number of original objects in its cluster
          is plotted first.

        Note, ``distance_sort`` and ``count_sort`` cannot both be True.
    distance_sort : str or bool, optional
        For each node n, the order (visually, from left-to-right) n's
        two descendent links are plotted is determined by this
        parameter, which can be any of the following values:

        ``False``
          Nothing is done.

        ``'ascending'`` or ``True``
          The child with the minimum distance between its direct descendents is
          plotted first.

        ``'descending'``
          The child with the maximum distance between its direct descendents is
          plotted first.

        Note ``distance_sort`` and ``count_sort`` cannot both be True.
    show_leaf_counts : bool, optional
         When True, leaf nodes representing :math:`k>1` original
         observation are labeled with the number of observations they
         contain in parentheses.
    no_plot : bool, optional
        When True, the final rendering is not performed. This is
        useful if only the data structures computed for the rendering
        are needed or if matplotlib is not available.
    no_labels : bool, optional
        When True, no labels appear next to the leaf nodes in the
        rendering of the dendrogram.
    leaf_rotation : double, optional
        Specifies the angle (in degrees) to rotate the leaf
        labels. When unspecified, the rotation is based on the number of
        nodes in the dendrogram (default is 0).
    leaf_font_size : int, optional
        Specifies the font size (in points) of the leaf labels. When
        unspecified, the size based on the number of nodes in the
        dendrogram.
    leaf_label_func : lambda or function, optional
        When ``leaf_label_func`` is a callable function, for each
        leaf with cluster index :math:`k < 2n-1`. The function
        is expected to return a string with the label for the
        leaf.

        Indices :math:`k < n` correspond to original observations
        while indices :math:`k \\geq n` correspond to non-singleton
        clusters.

        For example, to label singletons with their node id and
        non-singletons with their id, count, and inconsistency
        coefficient, simply do::

            # First define the leaf label function.
            def llf(id):
                if id < n:
                    return str(id)
                else:
                    return '[%d %d %1.2f]' % (id, count, R[n-id,3])

            # The text for the leaf nodes is going to be big so force
            # a rotation of 90 degrees.
            dendrogram(Z, leaf_label_func=llf, leaf_rotation=90)

            # leaf_label_func can also be used together with ``truncate_mode``,
            # in which case you will get your leaves labeled after truncation:
            dendrogram(Z, leaf_label_func=llf, leaf_rotation=90,
                       truncate_mode='level', p=2)

    show_contracted : bool, optional
        When True the heights of non-singleton nodes contracted
        into a leaf node are plotted as crosses along the link
        connecting that leaf node.  This really is only useful when
        truncation is used (see ``truncate_mode`` parameter).
    link_color_func : callable, optional
        If given, `link_color_function` is called with each non-singleton id
        corresponding to each U-shaped link it will paint. The function is
        expected to return the color to paint the link, encoded as a matplotlib
        color string code. For example::

            dendrogram(Z, link_color_func=lambda k: colors[k])

        colors the direct links below each untruncated non-singleton node
        ``k`` using ``colors[k]``.
    ax : matplotlib Axes instance, optional
        If None and `no_plot` is not True, the dendrogram will be plotted
        on the current axes.  Otherwise if `no_plot` is not True the
        dendrogram will be plotted on the given ``Axes`` instance. This can be
        useful if the dendrogram is part of a more complex figure.
    above_threshold_color : str, optional
        This matplotlib color string sets the color of the links above the
        color_threshold. The default is ``'C0'``.

    Returns
    -------
    R : dict
        A dictionary of data structures computed to render the
        dendrogram. Its has the following keys:

        ``'color_list'``
          A list of color names. The k'th element represents the color of the
          k'th link.

        ``'icoord'`` and ``'dcoord'``
          Each of them is a list of lists. Let ``icoord = [I1, I2, ..., Ip]``
          where ``Ik = [xk1, xk2, xk3, xk4]`` and ``dcoord = [D1, D2, ..., Dp]``
          where ``Dk = [yk1, yk2, yk3, yk4]``, then the k'th link painted is
          ``(xk1, yk1)`` - ``(xk2, yk2)`` - ``(xk3, yk3)`` - ``(xk4, yk4)``.

        ``'ivl'``
          A list of labels corresponding to the leaf nodes.

        ``'leaves'``
          For each i, ``H[i] == j``, cluster node ``j`` appears in position
          ``i`` in the left-to-right traversal of the leaves, where
          :math:`j < 2n-1` and :math:`i < n`. If ``j`` is less than ``n``, the
          ``i``-th leaf node corresponds to an original observation.
          Otherwise, it corresponds to a non-singleton cluster.

        ``'leaves_color_list'``
          A list of color names. The k'th element represents the color of the
          k'th leaf.

    See Also
    --------
    linkage, set_link_color_palette

    Notes
    -----
    It is expected that the distances in ``Z[:,2]`` be monotonic, otherwise
    crossings appear in the dendrogram.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.cluster import hierarchy
    >>> import matplotlib.pyplot as plt

    A very basic example:

    >>> ytdist = np.array([662., 877., 255., 412., 996., 295., 468., 268.,
    ...                    400., 754., 564., 138., 219., 869., 669.])
    >>> Z = hierarchy.linkage(ytdist, 'single')
    >>> plt.figure()
    >>> dn = hierarchy.dendrogram(Z)

    Now, plot in given axes, improve the color scheme and use both vertical and
    horizontal orientations:

    >>> hierarchy.set_link_color_palette(['m', 'c', 'y', 'k'])
    >>> fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    >>> dn1 = hierarchy.dendrogram(Z, ax=axes[0], above_threshold_color='y',
    ...                            orientation='top')
    >>> dn2 = hierarchy.dendrogram(Z, ax=axes[1],
    ...                            above_threshold_color='#bcbddc',
    ...                            orientation='right')
    >>> hierarchy.set_link_color_palette(None)  # reset to default after use
    >>> plt.show()

    """
    Z = as_xparray(Z, order='c')
    if orientation not in ['top', 'left', 'bottom', 'right']:
        raise ValueError("orientation must be one of 'top', 'left', 'bottom', or 'right'")
    if labels is not None:
        try:
            len_labels = len(labels)
        except (TypeError, AttributeError):
            len_labels = labels.shape[0]
        if Z.shape[0] + 1 != len_labels:
            raise ValueError('Dimensions of Z and labels must be consistent.')
    is_valid_linkage(Z, throw=True, name='Z')
    Zs = Z.shape
    n = Zs[0] + 1
    if isinstance(p, (int, float)):
        p = int(p)
    else:
        raise TypeError('The second argument must be a number')
    if truncate_mode not in ('lastp', 'mtica', 'level', 'none', None):
        raise ValueError('Invalid truncation mode.')
    if truncate_mode == 'lastp':
        if p > n or p == 0:
            p = n
    if truncate_mode == 'mtica':
        truncate_mode = 'level'
    if truncate_mode == 'level':
        if p <= 0:
            p = np.inf
    if get_leaves:
        lvs = []
    else:
        lvs = None
    icoord_list = []
    dcoord_list = []
    color_list = []
    current_color = [0]
    currently_below_threshold = [False]
    ivl = []
    if color_threshold is None or (isinstance(color_threshold, str) and color_threshold == 'default'):
        color_threshold = max(Z[:, 2]) * 0.7
    R = {'icoord': icoord_list, 'dcoord': dcoord_list, 'ivl': ivl, 'leaves': lvs, 'color_list': color_list}
    contraction_marks = [] if show_contracted else None
    _dendrogram_calculate_info(Z=Z, p=p, truncate_mode=truncate_mode, color_threshold=color_threshold, get_leaves=get_leaves, orientation=orientation, labels=labels, count_sort=count_sort, distance_sort=distance_sort, show_leaf_counts=show_leaf_counts, i=2 * n - 2, iv=0.0, ivl=ivl, n=n, icoord_list=icoord_list, dcoord_list=dcoord_list, lvs=lvs, current_color=current_color, color_list=color_list, currently_below_threshold=currently_below_threshold, leaf_label_func=leaf_label_func, contraction_marks=contraction_marks, link_color_func=link_color_func, above_threshold_color=above_threshold_color)
    if not no_plot:
        mh = max(Z[:, 2])
        _plot_dendrogram(icoord_list, dcoord_list, ivl, p, n, mh, orientation, no_labels, color_list, leaf_font_size=leaf_font_size, leaf_rotation=leaf_rotation, contraction_marks=contraction_marks, ax=ax, above_threshold_color=above_threshold_color)
    R['leaves_color_list'] = _get_leaves_color_list(R)
    return R