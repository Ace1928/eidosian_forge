import json
import decimal
import datetime
import warnings
from pathlib import Path
from plotly.io._utils import validate_coerce_fig_to_dict, validate_coerce_output_type
from _plotly_utils.optional_imports import get_module
from _plotly_utils.basevalidators import ImageUriValidator
from_json_plotly requires a string or bytes argument but received value of type {typ}
def clean_to_json_compatible(obj, **kwargs):
    if isinstance(obj, (int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: clean_to_json_compatible(v, **kwargs) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        if obj:
            return [clean_to_json_compatible(v, **kwargs) for v in obj]
    numpy_allowed = kwargs.get('numpy_allowed', False)
    datetime_allowed = kwargs.get('datetime_allowed', False)
    modules = kwargs.get('modules', {})
    sage_all = modules['sage_all']
    np = modules['np']
    pd = modules['pd']
    image = modules['image']
    if sage_all is not None:
        if obj in sage_all.RR:
            return float(obj)
        elif obj in sage_all.ZZ:
            return int(obj)
    if np is not None:
        if obj is np.ma.core.masked:
            return float('nan')
        elif isinstance(obj, np.ndarray):
            if numpy_allowed and obj.dtype.kind in ('b', 'i', 'u', 'f'):
                return np.ascontiguousarray(obj)
            elif obj.dtype.kind == 'M':
                return np.datetime_as_string(obj).tolist()
            elif obj.dtype.kind == 'U':
                return obj.tolist()
            elif obj.dtype.kind == 'O':
                obj = obj.tolist()
        elif isinstance(obj, np.datetime64):
            return str(obj)
    if pd is not None:
        if obj is pd.NaT:
            return None
        elif isinstance(obj, (pd.Series, pd.DatetimeIndex)):
            if numpy_allowed and obj.dtype.kind in ('b', 'i', 'u', 'f'):
                return np.ascontiguousarray(obj.values)
            elif obj.dtype.kind == 'M':
                if isinstance(obj, pd.Series):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', FutureWarning)
                        dt_values = np.array(obj.dt.to_pydatetime()).tolist()
                else:
                    dt_values = obj.to_pydatetime().tolist()
                if not datetime_allowed:
                    for i in range(len(dt_values)):
                        dt_values[i] = dt_values[i].isoformat()
                return dt_values
    try:
        obj = obj.to_pydatetime()
    except (TypeError, AttributeError):
        pass
    if not datetime_allowed:
        try:
            return obj.isoformat()
        except (TypeError, AttributeError):
            pass
    elif isinstance(obj, datetime.datetime):
        return obj
    try:
        return obj.tolist()
    except AttributeError:
        pass
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if image is not None and isinstance(obj, image.Image):
        return ImageUriValidator.pil_image_to_uri(obj)
    try:
        obj = obj.to_plotly_json()
    except AttributeError:
        pass
    if isinstance(obj, dict):
        return {k: clean_to_json_compatible(v, **kwargs) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        if obj:
            return [clean_to_json_compatible(v, **kwargs) for v in obj]
    return obj