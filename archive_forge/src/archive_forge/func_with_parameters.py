import inspect
import logging
import types
from typing import Any, Callable, Dict, Optional, Type, Union, TYPE_CHECKING
import ray
from ray.tune.execution.placement_groups import (
from ray.air.config import ScalingConfig
from ray.tune.registry import _ParameterRegistry
from ray.tune.utils import _detect_checkpoint_function
from ray.util.annotations import PublicAPI
@PublicAPI(stability='beta')
def with_parameters(trainable: Union[Type['Trainable'], Callable], **kwargs):
    """Wrapper for trainables to pass arbitrary large data objects.

    This wrapper function will store all passed parameters in the Ray
    object store and retrieve them when calling the function. It can thus
    be used to pass arbitrary data, even datasets, to Tune trainables.

    This can also be used as an alternative to ``functools.partial`` to pass
    default arguments to trainables.

    When used with the function API, the trainable function is called with
    the passed parameters as keyword arguments. When used with the class API,
    the ``Trainable.setup()`` method is called with the respective kwargs.

    If the data already exists in the object store (are instances of
    ObjectRef), using ``tune.with_parameters()`` is not necessary. You can
    instead pass the object refs to the training function via the ``config``
    or use Python partials.

    Args:
        trainable: Trainable to wrap.
        **kwargs: parameters to store in object store.

    Function API example:

    .. code-block:: python

        from ray import train, tune

        def train_fn(config, data=None):
            for sample in data:
                loss = update_model(sample)
                train.report(loss=loss)

        data = HugeDataset(download=True)

        tuner = Tuner(
            tune.with_parameters(train_fn, data=data),
            # ...
        )
        tuner.fit()

    Class API example:

    .. code-block:: python

        from ray import tune

        class MyTrainable(tune.Trainable):
            def setup(self, config, data=None):
                self.data = data
                self.iter = iter(self.data)
                self.next_sample = next(self.iter)

            def step(self):
                loss = update_model(self.next_sample)
                try:
                    self.next_sample = next(self.iter)
                except StopIteration:
                    return {"loss": loss, done: True}
                return {"loss": loss}

        data = HugeDataset(download=True)

        tuner = Tuner(
            tune.with_parameters(MyTrainable, data=data),
            # ...
        )
    """
    from ray.tune.trainable import Trainable
    if not callable(trainable) or (inspect.isclass(trainable) and (not issubclass(trainable, Trainable))):
        raise ValueError(f'`tune.with_parameters() only works with function trainables or classes that inherit from `tune.Trainable()`. Got type: {type(trainable)}.')
    parameter_registry = _ParameterRegistry()
    ray._private.worker._post_init_hooks.append(parameter_registry.flush)
    prefix = f'{str(trainable)}_'
    for k, v in kwargs.items():
        parameter_registry.put(prefix + k, v)
    trainable_name = getattr(trainable, '__name__', 'tune_with_parameters')
    keys = set(kwargs.keys())
    if inspect.isclass(trainable):

        class _Inner(trainable):

            def setup(self, config):
                setup_kwargs = {}
                for k in keys:
                    setup_kwargs[k] = parameter_registry.get(prefix + k)
                super(_Inner, self).setup(config, **setup_kwargs)
        trainable_with_params = _Inner
    else:
        if _detect_checkpoint_function(trainable, partial=True):
            from ray.tune.trainable.function_trainable import _CHECKPOINT_DIR_ARG_DEPRECATION_MSG
            raise DeprecationWarning(_CHECKPOINT_DIR_ARG_DEPRECATION_MSG)

        def inner(config):
            fn_kwargs = {}
            for k in keys:
                fn_kwargs[k] = parameter_registry.get(prefix + k)
            return trainable(config, **fn_kwargs)
        trainable_with_params = inner
        if hasattr(trainable, '__mixins__'):
            trainable_with_params.__mixins__ = trainable.__mixins__
        if hasattr(trainable, '_resources'):
            trainable_with_params._resources = trainable._resources
    trainable_with_params.__name__ = trainable_name
    return trainable_with_params