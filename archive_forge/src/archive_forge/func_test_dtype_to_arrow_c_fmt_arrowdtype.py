import numpy as np
import pytest
import pandas as pd
from pandas.core.interchange.utils import dtype_to_arrow_c_fmt
@pytest.mark.parametrize('pa_dtype, args_kwargs, c_string', [['null', {}, 'n'], ['bool_', {}, 'b'], ['uint8', {}, 'C'], ['uint16', {}, 'S'], ['uint32', {}, 'I'], ['uint64', {}, 'L'], ['int8', {}, 'c'], ['int16', {}, 'S'], ['int32', {}, 'i'], ['int64', {}, 'l'], ['float16', {}, 'e'], ['float32', {}, 'f'], ['float64', {}, 'g'], ['string', {}, 'u'], ['binary', {}, 'z'], ['time32', ('s',), 'tts'], ['time32', ('ms',), 'ttm'], ['time64', ('us',), 'ttu'], ['time64', ('ns',), 'ttn'], ['date32', {}, 'tdD'], ['date64', {}, 'tdm'], ['timestamp', {'unit': 's'}, 'tss:'], ['timestamp', {'unit': 'ms'}, 'tsm:'], ['timestamp', {'unit': 'us'}, 'tsu:'], ['timestamp', {'unit': 'ns'}, 'tsn:'], ['timestamp', {'unit': 'ns', 'tz': 'UTC'}, 'tsn:UTC'], ['duration', ('s',), 'tDs'], ['duration', ('ms',), 'tDm'], ['duration', ('us',), 'tDu'], ['duration', ('ns',), 'tDn'], ['decimal128', {'precision': 4, 'scale': 2}, 'd:4,2']])
def test_dtype_to_arrow_c_fmt_arrowdtype(pa_dtype, args_kwargs, c_string):
    pa = pytest.importorskip('pyarrow')
    if not args_kwargs:
        pa_type = getattr(pa, pa_dtype)()
    elif isinstance(args_kwargs, tuple):
        pa_type = getattr(pa, pa_dtype)(*args_kwargs)
    else:
        pa_type = getattr(pa, pa_dtype)(**args_kwargs)
    arrow_type = pd.ArrowDtype(pa_type)
    assert dtype_to_arrow_c_fmt(arrow_type) == c_string