from typing import Any, Callable
Module to facilitate adding hooks to wandb actions.

Usage:
    import trigger
    trigger.register('on_something', func)
    trigger.call('on_something', *args, **kwargs)
    trigger.unregister('on_something', func)
