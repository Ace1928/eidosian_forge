from toolz import curried
import uuid
from weakref import WeakValueDictionary
from typing import (
from altair.utils._importers import import_vegafusion
from altair.utils.core import DataFrameLike
from altair.utils.data import DataType, ToValuesReturnType, MaxRowsError
from altair.vegalite.data import default_data_transformer
def compile_with_vegafusion(vegalite_spec: dict) -> dict:
    """Compile a Vega-Lite spec to Vega and pre-transform with VegaFusion

    Note: This function should only be called on a Vega-Lite spec
    that was generated with the "vegafusion" data transformer enabled.
    In particular, this spec may contain references to extract datasets
    using table:// prefixed URLs.

    Parameters
    ----------
    vegalite_spec: dict
        A Vega-Lite spec that was generated from an Altair chart with
        the "vegafusion" data transformer enabled

    Returns
    -------
    dict
        A Vega spec that has been pre-transformed by VegaFusion
    """
    from altair import vegalite_compilers, data_transformers
    vf = import_vegafusion()
    compiler = vegalite_compilers.get()
    if compiler is None:
        raise ValueError('No active vega-lite compiler plugin found')
    vega_spec = compiler(vegalite_spec)
    inline_tables = get_inline_tables(vega_spec)
    row_limit = data_transformers.options.get('max_rows', None)
    transformed_vega_spec, warnings = vf.runtime.pre_transform_spec(vega_spec, vf.get_local_tz(), inline_datasets=inline_tables, row_limit=row_limit)
    handle_row_limit_exceeded(row_limit, warnings)
    return transformed_vega_spec