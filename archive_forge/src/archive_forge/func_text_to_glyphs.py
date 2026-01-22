from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def text_to_glyphs(self, x, y, text, with_clusters):
    """Converts a string of text to a list of glyphs,
        optionally with cluster mapping,
        that can be used to render later using this scaled font.

        The output values can be readily passed to
        :meth:`Context.show_text_glyphs`, :meth:`Context.show_glyphs`
        or related methods,
        assuming that the exact same :class:`ScaledFont`
        is used for the operation.

        :type x: float
        :type y: float
        :type with_clusters: bool
        :param x: X position to place first glyph.
        :param y: Y position to place first glyph.
        :param text: The text to convert, as an Unicode or UTF-8 string.
        :param with_clusters: Whether to compute the cluster mapping.
        :returns:
            A ``(glyphs, clusters, clusters_flags)`` tuple
            if ``with_clusters`` is true, otherwise just ``glyphs``.
            See :meth:`Context.show_text_glyphs` for the data structure.

        .. note::

            This method is part of
            what the cairo designers call the "toy" text API.
            It is convenient for short demos and simple programs,
            but it is not expected to be adequate
            for serious text-using applications.
            See :ref:`fonts` for details
            and :meth:`Context.show_glyphs`
            for the "real" text display API in cairo.

        """
    glyphs = ffi.new('cairo_glyph_t **', ffi.NULL)
    num_glyphs = ffi.new('int *')
    if with_clusters:
        clusters = ffi.new('cairo_text_cluster_t **', ffi.NULL)
        num_clusters = ffi.new('int *')
        cluster_flags = ffi.new('cairo_text_cluster_flags_t *')
    else:
        clusters = ffi.NULL
        num_clusters = ffi.NULL
        cluster_flags = ffi.NULL
    status = cairo.cairo_scaled_font_text_to_glyphs(self._pointer, x, y, _encode_string(text), -1, glyphs, num_glyphs, clusters, num_clusters, cluster_flags)
    glyphs = ffi.gc(glyphs[0], _keepref(cairo, cairo.cairo_glyph_free))
    if with_clusters:
        clusters = ffi.gc(clusters[0], _keepref(cairo, cairo.cairo_text_cluster_free))
    _check_status(status)
    glyphs = [(glyph.index, glyph.x, glyph.y) for i in range(num_glyphs[0]) for glyph in [glyphs[i]]]
    if with_clusters:
        clusters = [(cluster.num_bytes, cluster.num_glyphs) for i in range(num_clusters[0]) for cluster in [clusters[i]]]
        return (glyphs, clusters, cluster_flags[0])
    else:
        return glyphs