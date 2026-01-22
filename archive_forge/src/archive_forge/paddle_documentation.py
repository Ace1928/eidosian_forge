import os
from typing import Dict, Optional, Union
import numpy as np
import paddle
from safetensors import numpy

    Loads a safetensors file into paddle format.

    Args:
        filename (`str`, or `os.PathLike`)):
            The name of the file which contains the tensors
        device (`Dict[str, any]`, *optional*, defaults to `cpu`):
            The device where the tensors need to be located after load.
            available options are all regular paddle device locations

    Returns:
        `Dict[str, paddle.Tensor]`: dictionary that contains name as key, value as `paddle.Tensor`

    Example:

    ```python
    from safetensors.paddle import load_file

    file_path = "./my_folder/bert.safetensors"
    loaded = load_file(file_path)
    ```
    