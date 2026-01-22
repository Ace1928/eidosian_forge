from typing import Callable, Optional
from pytorch_lightning.callbacks.callback import Callback
Create a simple callback on the fly using lambda functions.

    Args:
        **kwargs: hooks supported by :class:`~pytorch_lightning.callbacks.callback.Callback`

    Example::

        >>> from pytorch_lightning import Trainer
        >>> from pytorch_lightning.callbacks import LambdaCallback
        >>> trainer = Trainer(callbacks=[LambdaCallback(setup=lambda *args: print('setup'))])

    