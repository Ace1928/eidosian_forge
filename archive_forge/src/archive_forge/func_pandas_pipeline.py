from __future__ import annotations
import pandas as pd
from datashader.core import bypixel
from datashader.compiler import compile_components
from datashader.glyphs.points import _PointLike, _GeometryLike
from datashader.glyphs.area import _AreaToLineLike
from datashader.glyphs.line import LinesXarrayCommonX
from datashader.utils import Dispatcher
@bypixel.pipeline.register(pd.DataFrame)
def pandas_pipeline(df, schema, canvas, glyph, summary, *, antialias=False):
    return glyph_dispatch(glyph, df, schema, canvas, summary, antialias=antialias)