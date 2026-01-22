import numpy as np
from ._ckdtree import cKDTree, cKDTreeNode
def count_neighbors(self, other, r, p=2.0, weights=None, cumulative=True):
    """Count how many nearby pairs can be formed.

        Count the number of pairs ``(x1,x2)`` can be formed, with ``x1`` drawn
        from ``self`` and ``x2`` drawn from ``other``, and where
        ``distance(x1, x2, p) <= r``.

        Data points on ``self`` and ``other`` are optionally weighted by the
        ``weights`` argument. (See below)

        This is adapted from the "two-point correlation" algorithm described by
        Gray and Moore [1]_.  See notes for further discussion.

        Parameters
        ----------
        other : KDTree
            The other tree to draw points from, can be the same tree as self.
        r : float or one-dimensional array of floats
            The radius to produce a count for. Multiple radii are searched with
            a single tree traversal.
            If the count is non-cumulative(``cumulative=False``), ``r`` defines
            the edges of the bins, and must be non-decreasing.
        p : float, optional
            1<=p<=infinity.
            Which Minkowski p-norm to use.
            Default 2.0.
            A finite large p may cause a ValueError if overflow can occur.
        weights : tuple, array_like, or None, optional
            If None, the pair-counting is unweighted.
            If given as a tuple, weights[0] is the weights of points in
            ``self``, and weights[1] is the weights of points in ``other``;
            either can be None to indicate the points are unweighted.
            If given as an array_like, weights is the weights of points in
            ``self`` and ``other``. For this to make sense, ``self`` and
            ``other`` must be the same tree. If ``self`` and ``other`` are two
            different trees, a ``ValueError`` is raised.
            Default: None

            .. versionadded:: 1.6.0
        cumulative : bool, optional
            Whether the returned counts are cumulative. When cumulative is set
            to ``False`` the algorithm is optimized to work with a large number
            of bins (>10) specified by ``r``. When ``cumulative`` is set to
            True, the algorithm is optimized to work with a small number of
            ``r``. Default: True

            .. versionadded:: 1.6.0

        Returns
        -------
        result : scalar or 1-D array
            The number of pairs. For unweighted counts, the result is integer.
            For weighted counts, the result is float.
            If cumulative is False, ``result[i]`` contains the counts with
            ``(-inf if i == 0 else r[i-1]) < R <= r[i]``

        Notes
        -----
        Pair-counting is the basic operation used to calculate the two point
        correlation functions from a data set composed of position of objects.

        Two point correlation function measures the clustering of objects and
        is widely used in cosmology to quantify the large scale structure
        in our Universe, but it may be useful for data analysis in other fields
        where self-similar assembly of objects also occur.

        The Landy-Szalay estimator for the two point correlation function of
        ``D`` measures the clustering signal in ``D``. [2]_

        For example, given the position of two sets of objects,

        - objects ``D`` (data) contains the clustering signal, and

        - objects ``R`` (random) that contains no signal,

        .. math::

             \\xi(r) = \\frac{<D, D> - 2 f <D, R> + f^2<R, R>}{f^2<R, R>},

        where the brackets represents counting pairs between two data sets
        in a finite bin around ``r`` (distance), corresponding to setting
        `cumulative=False`, and ``f = float(len(D)) / float(len(R))`` is the
        ratio between number of objects from data and random.

        The algorithm implemented here is loosely based on the dual-tree
        algorithm described in [1]_. We switch between two different
        pair-cumulation scheme depending on the setting of ``cumulative``.
        The computing time of the method we use when for
        ``cumulative == False`` does not scale with the total number of bins.
        The algorithm for ``cumulative == True`` scales linearly with the
        number of bins, though it is slightly faster when only
        1 or 2 bins are used. [5]_.

        As an extension to the naive pair-counting,
        weighted pair-counting counts the product of weights instead
        of number of pairs.
        Weighted pair-counting is used to estimate marked correlation functions
        ([3]_, section 2.2),
        or to properly calculate the average of data per distance bin
        (e.g. [4]_, section 2.1 on redshift).

        .. [1] Gray and Moore,
               "N-body problems in statistical learning",
               Mining the sky, 2000,
               https://arxiv.org/abs/astro-ph/0012333

        .. [2] Landy and Szalay,
               "Bias and variance of angular correlation functions",
               The Astrophysical Journal, 1993,
               http://adsabs.harvard.edu/abs/1993ApJ...412...64L

        .. [3] Sheth, Connolly and Skibba,
               "Marked correlations in galaxy formation models",
               Arxiv e-print, 2005,
               https://arxiv.org/abs/astro-ph/0511773

        .. [4] Hawkins, et al.,
               "The 2dF Galaxy Redshift Survey: correlation functions,
               peculiar velocities and the matter density of the Universe",
               Monthly Notices of the Royal Astronomical Society, 2002,
               http://adsabs.harvard.edu/abs/2003MNRAS.346...78H

        .. [5] https://github.com/scipy/scipy/pull/5647#issuecomment-168474926

        Examples
        --------
        You can count neighbors number between two kd-trees within a distance:

        >>> import numpy as np
        >>> from scipy.spatial import KDTree
        >>> rng = np.random.default_rng()
        >>> points1 = rng.random((5, 2))
        >>> points2 = rng.random((5, 2))
        >>> kd_tree1 = KDTree(points1)
        >>> kd_tree2 = KDTree(points2)
        >>> kd_tree1.count_neighbors(kd_tree2, 0.2)
        1

        This number is same as the total pair number calculated by
        `query_ball_tree`:

        >>> indexes = kd_tree1.query_ball_tree(kd_tree2, r=0.2)
        >>> sum([len(i) for i in indexes])
        1

        """
    return super().count_neighbors(other, r, p, weights, cumulative)