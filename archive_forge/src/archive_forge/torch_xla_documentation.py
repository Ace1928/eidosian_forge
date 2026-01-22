import importlib.metadata
import subprocess
import sys

    Helper function to install appropriate xla wheels based on the `torch` version in Google Colaboratory.

    Args:
        upgrade (`bool`, *optional*, defaults to `False`):
            Whether to upgrade `torch` and install the latest `torch_xla` wheels.

    Example:

    ```python
    >>> from accelerate.utils import install_xla

    >>> install_xla(upgrade=True)
    ```
    