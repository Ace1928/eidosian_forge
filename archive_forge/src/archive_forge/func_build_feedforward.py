from pathlib import Path
from typing import Any, Callable, Dict, Set, Union
from xformers.utils import (
from .base import Feedforward, FeedforwardConfig  # noqa
from .mlp import MLP  # noqa
import_all_modules(str(Path(__file__).parent), "xformers.components.feedforward")
def build_feedforward(config: Union[Dict[str, Any], FeedforwardConfig]):
    """Builds a feedforward from a config.

    This assumes a 'name' key in the config which is used to determine what
    attention class to instantiate. For instance, a config `{"name": "my_feedforward",
    "foo": "bar"}` will find a class that was registered as "my_feedforward"
    (see :func:`register_feedforward`) and call .from_config on it."""
    if not isinstance(config, FeedforwardConfig):
        config_instance = generate_matching_config(config, FEEDFORWARD_REGISTRY[config['name']].config)
    else:
        config_instance = config
    return FEEDFORWARD_REGISTRY[config_instance.name].constructor.from_config(config_instance)