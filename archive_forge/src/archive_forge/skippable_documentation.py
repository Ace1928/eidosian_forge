from typing import (
from torch import Tensor, nn
from ..microbatch import Batch
from .namespace import Namespace
from .tracker import current_skip_tracker
Perform the forward propagation.

        :class:`stash` or :class:`pop` commands will be handled by portals
        silently. The portals won't be exposed to users.

        Raises:
            RuntimeError:
                illegal 'stash' or 'pop' is found.

        