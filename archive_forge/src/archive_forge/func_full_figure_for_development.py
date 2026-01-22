import os
import json
from pathlib import Path
import plotly
from plotly.io._utils import validate_coerce_fig_to_dict
def full_figure_for_development(fig, warn=True, as_dict=False):
    """
    Compute default values for all attributes not specified in the input figure and
    returns the output as a "full" figure. This function calls Plotly.js via Kaleido
    to populate unspecified attributes. This function is intended for interactive use
    during development to learn more about how Plotly.js computes default values and is
    not generally necessary or recommended for production use.

    Parameters
    ----------
    fig:
        Figure object or dict representing a figure

    warn: bool
        If False, suppress warnings about not using this in production.

    as_dict: bool
        If True, output is a dict with some keys that go.Figure can't parse.
        If False, output is a go.Figure with unparseable keys skipped.

    Returns
    -------
    plotly.graph_objects.Figure or dict
        The full figure
    """
    if scope is None:
        raise ValueError('\nFull figure generation requires the kaleido package,\nwhich can be installed using pip:\n    $ pip install -U kaleido\n')
    if warn:
        import warnings
        warnings.warn('full_figure_for_development is not recommended or necessary for production use in most circumstances. \nTo suppress this warning, set warn=False')
    fig = json.loads(scope.transform(fig, format='json').decode('utf-8'))
    if as_dict:
        return fig
    else:
        import plotly.graph_objects as go
        return go.Figure(fig, skip_invalid=True)