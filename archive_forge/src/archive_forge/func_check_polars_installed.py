from typing import TYPE_CHECKING, List
def check_polars_installed():
    try:
        global pl
        import polars as pl
    except ImportError:
        raise ImportError('polars not installed. Install with `pip install polars` or set `DataContext.use_polars = False` to fall back to pyarrow')