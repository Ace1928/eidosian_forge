import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union

    # Inject file into template datapipe.pyi.in.

    TODO: The current implementation of this script only generates interfaces for built-in methods. To generate
          interface for user-defined DataPipes, consider changing `IterDataPipe.register_datapipe_as_function`.
    