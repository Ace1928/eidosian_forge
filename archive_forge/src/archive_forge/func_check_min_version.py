from huggingface_hub import get_full_repo_name  # for backward compatibility
from huggingface_hub.constants import HF_HUB_DISABLE_TELEMETRY as DISABLE_TELEMETRY  # for backward compatibility
from packaging import version
from .. import __version__
from .backbone_utils import BackboneConfigMixin, BackboneMixin
from .constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD
from .doc import (
from .generic import (
from .hub import (
from .import_utils import (
from .peft_utils import (
def check_min_version(min_version):
    if version.parse(__version__) < version.parse(min_version):
        if 'dev' in min_version:
            error_message = 'This example requires a source install from HuggingFace Transformers (see `https://huggingface.co/docs/transformers/installation#install-from-source`),'
        else:
            error_message = f'This example requires a minimum version of {min_version},'
        error_message += f' but the version found is {__version__}.\n'
        raise ImportError(error_message + 'Check out https://github.com/huggingface/transformers/tree/main/examples#important-note for the examples corresponding to other versions of HuggingFace Transformers.')