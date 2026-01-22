from __future__ import annotations
import logging  # isort:skip
import importlib
from docutils import nodes
from sphinx.errors import SphinxError
from . import PARALLEL_SAFE
def bokeh_dataframe(name, rawtext, text, lineno, inliner, options=None, content=None):
    """Generate an inline visual representation of a single color palette.

    If the HTML representation of the dataframe can not be created, a
    SphinxError is raised to terminate the build.

    For details on the arguments to this function, consult the Docutils docs:

    http://docutils.sourceforge.net/docs/howto/rst-roles.html#define-the-role-function

    """
    import pandas as pd
    module_name, df_name = text.rsplit('.', 1)
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        raise SphinxError(f"Unable to generate HTML table for {df_name}: couldn't import module {module_name}")
    df = getattr(module, df_name, None)
    if df is None:
        raise SphinxError(f'Unable to generate HTML table for {df_name}: no Dataframe {df_name} in module {module_name}')
    if not isinstance(df, pd.DataFrame):
        raise SphinxError(f'{text!r} is not a pandas Dataframe')
    node = nodes.raw('', df.head().to_html(), format='html')
    return ([node], [])