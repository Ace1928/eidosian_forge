from . import Request
from ..typing import ArrayLike
import numpy as np
from typing import Optional, Dict, Any, Tuple, Union, List, Iterator
from dataclasses import dataclass
Close the ImageResource.

        This method allows a plugin to behave similar to the python built-in ``open``::

            image_file = my_plugin(Request, "r")
            ...
            image_file.close()

        It is used by the context manager and deconstructor below to avoid leaking
        ImageResources. If the plugin has no other cleanup to do it doesn't have
        to overwrite this method itself and can rely on the implementation
        below.

        