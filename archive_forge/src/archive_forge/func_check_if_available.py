from functools import wraps
import transformers
from packaging import version
def check_if_available(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not are_fx_features_available():
            raise ImportError(f'Found an incompatible version of transformers. Found version {transformers_version}, but only {_TRANSFORMERS_MIN_VERSION} and above are supported.')
        return func(*args, **kwargs)
    return wrapper