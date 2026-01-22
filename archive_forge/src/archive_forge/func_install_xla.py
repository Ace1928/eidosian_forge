import importlib.metadata
import subprocess
import sys
def install_xla(upgrade: bool=False):
    """
    Helper function to install appropriate xla wheels based on the `torch` version in Google Colaboratory.

    Args:
        upgrade (`bool`, *optional*, defaults to `False`):
            Whether to upgrade `torch` and install the latest `torch_xla` wheels.

    Example:

    ```python
    >>> from accelerate.utils import install_xla

    >>> install_xla(upgrade=True)
    ```
    """
    in_colab = False
    if 'IPython' in sys.modules:
        in_colab = 'google.colab' in str(sys.modules['IPython'].get_ipython())
    if in_colab:
        if upgrade:
            torch_install_cmd = ['pip', 'install', '-U', 'torch']
            subprocess.run(torch_install_cmd, check=True)
        torch_version = importlib.metadata.version('torch')
        torch_version_trunc = torch_version[:torch_version.rindex('.')]
        xla_wheel = f'https://storage.googleapis.com/tpu-pytorch/wheels/colab/torch_xla-{torch_version_trunc}-cp37-cp37m-linux_x86_64.whl'
        xla_install_cmd = ['pip', 'install', xla_wheel]
        subprocess.run(xla_install_cmd, check=True)
    else:
        raise RuntimeError('`install_xla` utility works only on google colab.')