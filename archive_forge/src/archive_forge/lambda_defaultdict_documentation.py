from collections import defaultdict
from typing import Any, Callable
Initializes a LambdaDefaultDict instance.

        Args:
            default_factory: The default factory callable, taking a string (key)
                and returning the default value to use for that key.
        