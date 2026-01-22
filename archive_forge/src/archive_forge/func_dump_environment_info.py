import importlib.metadata
import platform
import sys
import warnings
from typing import Any, Dict
from .. import __version__, constants
def dump_environment_info() -> Dict[str, Any]:
    """Dump information about the machine to help debugging issues.

    Similar helper exist in:
    - `datasets` (https://github.com/huggingface/datasets/blob/main/src/datasets/commands/env.py)
    - `diffusers` (https://github.com/huggingface/diffusers/blob/main/src/diffusers/commands/env.py)
    - `transformers` (https://github.com/huggingface/transformers/blob/main/src/transformers/commands/env.py)
    """
    from huggingface_hub import get_token, whoami
    from huggingface_hub.utils import list_credential_helpers
    token = get_token()
    info: Dict[str, Any] = {'huggingface_hub version': get_hf_hub_version(), 'Platform': platform.platform(), 'Python version': get_python_version()}
    try:
        shell_class = get_ipython().__class__
        info['Running in iPython ?'] = 'Yes'
        info['iPython shell'] = shell_class.__name__
    except NameError:
        info['Running in iPython ?'] = 'No'
    info['Running in notebook ?'] = 'Yes' if is_notebook() else 'No'
    info['Running in Google Colab ?'] = 'Yes' if is_google_colab() else 'No'
    info['Token path ?'] = constants.HF_TOKEN_PATH
    info['Has saved token ?'] = token is not None
    if token is not None:
        try:
            info['Who am I ?'] = whoami()['name']
        except Exception:
            pass
    try:
        info['Configured git credential helpers'] = ', '.join(list_credential_helpers())
    except Exception:
        pass
    info['FastAI'] = get_fastai_version()
    info['Tensorflow'] = get_tf_version()
    info['Torch'] = get_torch_version()
    info['Jinja2'] = get_jinja_version()
    info['Graphviz'] = get_graphviz_version()
    info['Pydot'] = get_pydot_version()
    info['Pillow'] = get_pillow_version()
    info['hf_transfer'] = get_hf_transfer_version()
    info['gradio'] = get_gradio_version()
    info['tensorboard'] = get_tensorboard_version()
    info['numpy'] = get_numpy_version()
    info['pydantic'] = get_pydantic_version()
    info['aiohttp'] = get_aiohttp_version()
    info['ENDPOINT'] = constants.ENDPOINT
    info['HF_HUB_CACHE'] = constants.HF_HUB_CACHE
    info['HF_ASSETS_CACHE'] = constants.HF_ASSETS_CACHE
    info['HF_TOKEN_PATH'] = constants.HF_TOKEN_PATH
    info['HF_HUB_OFFLINE'] = constants.HF_HUB_OFFLINE
    info['HF_HUB_DISABLE_TELEMETRY'] = constants.HF_HUB_DISABLE_TELEMETRY
    info['HF_HUB_DISABLE_PROGRESS_BARS'] = constants.HF_HUB_DISABLE_PROGRESS_BARS
    info['HF_HUB_DISABLE_SYMLINKS_WARNING'] = constants.HF_HUB_DISABLE_SYMLINKS_WARNING
    info['HF_HUB_DISABLE_EXPERIMENTAL_WARNING'] = constants.HF_HUB_DISABLE_EXPERIMENTAL_WARNING
    info['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = constants.HF_HUB_DISABLE_IMPLICIT_TOKEN
    info['HF_HUB_ENABLE_HF_TRANSFER'] = constants.HF_HUB_ENABLE_HF_TRANSFER
    info['HF_HUB_ETAG_TIMEOUT'] = constants.HF_HUB_ETAG_TIMEOUT
    info['HF_HUB_DOWNLOAD_TIMEOUT'] = constants.HF_HUB_DOWNLOAD_TIMEOUT
    print('\nCopy-and-paste the text below in your GitHub issue.\n')
    print('\n'.join([f'- {prop}: {val}' for prop, val in info.items()]) + '\n')
    return info