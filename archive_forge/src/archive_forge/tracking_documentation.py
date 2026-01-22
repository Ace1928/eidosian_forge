from functools import wraps
from keras.src.backend.common.global_state import get_global_attribute
from keras.src.backend.common.global_state import set_global_attribute
from keras.src.utils import python_utils
Attribute tracker, used for e.g. Variable tracking.

    Monitors certain attribute types
    and put them in appropriate lists in case of a match.

    Also passively tracks certain mutable collections
    (dict, list) so that items added to them later
    still get tracked. This is done by wrapping these
    collections into an equivalent, tracking-aware object.

    Usage:

    ```python
    def __init__(self):
        self.tracker = Tracker(
            # Format: `name: (test_fn, store)`
            {
                "variables":
                    (lambda x: isinstance(x, Variable), self._variables),
                "metrics": (lambda x: isinstance(x, Metric), self._metrics),
                "layers": (lambda x: isinstance(x, Layer), self._layers),
            }
        )

    def __setattr__(self, name, value):
        if hasattr(self, "_tracker"):
            value = self._tracker.track(value)
        return super().__setattr__(name, value)
    ```
    