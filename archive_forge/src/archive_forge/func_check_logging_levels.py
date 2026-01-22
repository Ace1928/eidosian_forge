from typing import Optional, Tuple, Union
from typing_extensions import TypedDict
from pytorch_lightning.utilities.exceptions import MisconfigurationException
@classmethod
def check_logging_levels(cls, fx_name: str, on_step: bool, on_epoch: bool) -> None:
    """Check if the logging levels are allowed in the given hook."""
    fx_config = cls.functions[fx_name]
    assert fx_config is not None
    m = "You can't `self.log({}={})` inside `{}`, must be one of {}."
    if on_step not in fx_config['allowed_on_step']:
        msg = m.format('on_step', on_step, fx_name, fx_config['allowed_on_step'])
        raise MisconfigurationException(msg)
    if on_epoch not in fx_config['allowed_on_epoch']:
        msg = m.format('on_epoch', on_epoch, fx_name, fx_config['allowed_on_epoch'])
        raise MisconfigurationException(msg)