import copy
import importlib.metadata as importlib_metadata
import importlib.util
import weakref
from functools import partialmethod
from ..dependency_versions_check import dep_version_check
from ..utils import is_accelerate_available, is_torch_available, logging
def deepspeed_optim_sched(trainer, hf_deepspeed_config, args, num_training_steps, model_parameters):
    """
    A convenience wrapper that deals with optimizer and lr scheduler configuration.
    """
    from accelerate.utils import DummyOptim, DummyScheduler
    config = hf_deepspeed_config.config
    optimizer = None
    if 'optimizer' in config:
        if args.adafactor:
            raise ValueError('--adafactor was passed, but also found `optimizer` configured in the DeepSpeed config. Only one optimizer can be configured.')
        optimizer = DummyOptim(params=model_parameters)
    else:
        if hf_deepspeed_config.is_offload():
            logger.info('Detected ZeRO Offload and non-DeepSpeed optimizers: This combination should work as long as the custom optimizer has both CPU and GPU implementation (except LAMB)')
        optimizer = trainer.create_optimizer()
        config['zero_allow_untested_optimizer'] = True
    lr_scheduler = None
    if 'scheduler' in config:
        lr_scheduler = DummyScheduler(optimizer)
    elif isinstance(optimizer, DummyOptim):

        def _lr_scheduler_callable(optimizer):
            trainer_copy = copy.copy(trainer)
            trainer_copy.lr_scheduler = None
            lr_scheduler = trainer_copy.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)
            return lr_scheduler
        lr_scheduler = DummyScheduler(optimizer, lr_scheduler_callable=_lr_scheduler_callable)
    else:
        lr_scheduler = trainer.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)
    return (optimizer, lr_scheduler)