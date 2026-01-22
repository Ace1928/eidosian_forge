from __future__ import annotations
import numpy as np
import pandas as pd
import dask
import dask.array as da
import dask.dataframe as dd
from dask.base import tokenize, compute
from datashader.core import bypixel
from datashader.utils import apply
from datashader.compiler import compile_components
from datashader.glyphs import Glyph, LineAxis0
from datashader.utils import Dispatcher
@bypixel.pipeline.register(dd.DataFrame)
def dask_pipeline(df, schema, canvas, glyph, summary, *, antialias=False, cuda=False):
    dsk, name = glyph_dispatch(glyph, df, schema, canvas, summary, antialias=antialias, cuda=cuda)
    scheduler = dask.base.get_scheduler() or df.__dask_scheduler__
    if isinstance(dsk, da.Array):
        return da.compute(dsk, scheduler=scheduler)[0]
    keys = df.__dask_keys__()
    optimize = df.__dask_optimize__
    graph = df.__dask_graph__()
    dsk.update(optimize(graph, keys))
    return scheduler(dsk, name)