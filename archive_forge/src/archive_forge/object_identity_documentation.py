import collections
from typing import Any, Set
import weakref
Returns the referenced object.

    ```python
    x_ref = Reference(x)
    print(x is x_ref.deref())
    ==> True
    ```
    