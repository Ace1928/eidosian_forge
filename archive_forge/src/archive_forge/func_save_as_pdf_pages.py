from __future__ import annotations
import typing
from collections.abc import Sequence
from copy import copy, deepcopy
from io import BytesIO
from itertools import chain
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any, Dict, Iterable, Optional
from warnings import warn
from ._utils import (
from ._utils.context import plot_context
from ._utils.ipython import (
from .coords import coord_cartesian
from .exceptions import PlotnineError, PlotnineWarning
from .facets import facet_null
from .facets.layout import Layout
from .geoms.geom_blank import geom_blank
from .guides.guides import guides
from .iapi import mpl_save_view
from .layer import Layers
from .mapping.aes import aes, make_labels
from .options import get_option
from .scales.scales import Scales
from .themes.theme import theme, theme_get
def save_as_pdf_pages(plots: Iterable[ggplot], filename: Optional[str | Path]=None, path: str | None=None, verbose: bool=True, **kwargs: Any):
    """
    Save multiple [](`~plotnine.ggplot`) objects to a PDF file, one per page.

    Parameters
    ----------
    plots :
        Plot objects to write to file. `plots` may be either a
        collection such as a [](:class:`list`) or [](:class:`set`)

        ```python
        base_plot = ggplot(…)
        plots = [base_plot + ggtitle('%d of 3' % i) for i in range(1, 3)]
        save_as_pdf_pages(plots)
        ```

        or, a generator that yields [](`~plotnine.ggplot`) objects:

        ```python
        def myplots():
            for i in range(1, 3):
                yield ggplot(…) + ggtitle('%d of 3' % i)
        save_as_pdf_pages(myplots())
        ```
    filename :
        File name to write the plot to. If not specified, a name
        like “plotnine-save-<hash>.pdf” is used.
    path :
        Path to save plot to (if you just want to set path and
        not filename).
    verbose :
        If `True`, print the saving information.
    kwargs :
        Additional arguments to pass to
        [](:meth:`~matplotlib.figure.Figure.savefig`).

    Notes
    -----
    Using pandas [](:meth:`~pandas.DataFrame.groupby`) methods, tidy data
    can be "faceted" across pages:

    ```python
    from plotnine.data import mtcars

    def facet_pages(column)
        base_plot = [
            aes(x="wt", y="mpg", label="name"),
            geom_text(),
        ]
        for label, group_data in mtcars.groupby(column):
            yield ggplot(group_data) + base_plot + ggtitle(label)

    save_as_pdf_pages(facet_pages('cyl'))
    ```

    Unlike [](:meth:`~plotnine.ggplot.save`),
    [](:meth:`~plotnine.save_as_pdf_pages`)
    does not process arguments for `height` or `width`. To set the figure
    size, add [](`~plotnine.themes.themeable.figure_size`) to the theme
    for some or all of the objects in `plots`:

    ```python
    plot = ggplot(…)
    # The following are equivalent
    plot.save('filename.pdf', height=6, width=8)
    save_as_pdf_pages([plot + theme(figure_size=(8, 6))])
    ```
    """
    from matplotlib.backends.backend_pdf import PdfPages
    fig_kwargs = {'bbox_inches': 'tight'}
    fig_kwargs.update(kwargs)
    plots = iter(plots)
    if filename is None:
        peek = [next(plots)]
        plots = chain(peek, plots)
        filename = peek[0]._save_filename('pdf')
    if path:
        filename = Path(path) / filename
    if verbose:
        warn(f'Filename: {filename}', PlotnineWarning)
    with PdfPages(filename, keep_empty=False) as pdf:
        for plot in plots:
            fig = plot.draw()
            pdf.savefig(fig, **fig_kwargs)