import functools
import importlib.metadata
import importlib.util
import json
import numbers
import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import asdict, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional, Union
import numpy as np
from .. import __version__ as version
from ..utils import flatten_dict, is_datasets_available, is_pandas_available, is_torch_available, logging
from ..trainer_callback import ProgressCallback, TrainerCallback  # noqa: E402
from ..trainer_utils import PREFIX_CHECKPOINT_DIR, BestRun, IntervalStrategy  # noqa: E402
from ..training_args import ParallelMode  # noqa: E402
from ..utils import ENV_VARS_TRUE_VALUES, is_torch_tpu_available  # noqa: E402
def run_hp_search_ray(trainer, n_trials: int, direction: str, **kwargs) -> BestRun:
    import ray
    import ray.train

    def _objective(trial: dict, local_trainer):
        try:
            from transformers.utils.notebook import NotebookProgressCallback
            if local_trainer.pop_callback(NotebookProgressCallback):
                local_trainer.add_callback(ProgressCallback)
        except ModuleNotFoundError:
            pass
        local_trainer.objective = None
        checkpoint = ray.train.get_checkpoint()
        if checkpoint:
            local_trainer.objective = 'objective'
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_path = next(Path(checkpoint_dir).glob(f'{PREFIX_CHECKPOINT_DIR}*')).as_posix()
                local_trainer.train(resume_from_checkpoint=checkpoint_path, trial=trial)
        else:
            local_trainer.train(trial=trial)
        if getattr(local_trainer, 'objective', None) is None:
            metrics = local_trainer.evaluate()
            local_trainer.objective = local_trainer.compute_objective(metrics)
            metrics.update({'objective': local_trainer.objective, 'done': True})
            with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
                local_trainer._tune_save_checkpoint(checkpoint_dir=temp_checkpoint_dir)
                checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
                ray.train.report(metrics, checkpoint=checkpoint)
    if not trainer._memory_tracker.skip_memory_metrics:
        from ..trainer_utils import TrainerMemoryTracker
        logger.warning('Memory tracking for your Trainer is currently enabled. Automatically disabling the memory tracker since the memory tracker is not serializable.')
        trainer._memory_tracker = TrainerMemoryTracker(skip_memory_metrics=True)
    _tb_writer = trainer.pop_callback(TensorBoardCallback)
    trainer.model = None
    if 'resources_per_trial' not in kwargs:
        kwargs['resources_per_trial'] = {'cpu': 1}
        if trainer.args.n_gpu > 0:
            kwargs['resources_per_trial']['gpu'] = 1
        resource_msg = '1 CPU' + (' and 1 GPU' if trainer.args.n_gpu > 0 else '')
        logger.info(f'No `resources_per_trial` arg was passed into `hyperparameter_search`. Setting it to a default value of {resource_msg} for each trial.')
    gpus_per_trial = kwargs['resources_per_trial'].get('gpu', 0)
    trainer.args._n_gpu = gpus_per_trial
    if 'progress_reporter' not in kwargs:
        from ray.tune import CLIReporter
        kwargs['progress_reporter'] = CLIReporter(metric_columns=['objective'])
    if 'scheduler' in kwargs:
        from ray.tune.schedulers import ASHAScheduler, HyperBandForBOHB, MedianStoppingRule, PopulationBasedTraining
        if isinstance(kwargs['scheduler'], (ASHAScheduler, MedianStoppingRule, HyperBandForBOHB, PopulationBasedTraining)) and (not trainer.args.do_eval or trainer.args.evaluation_strategy == IntervalStrategy.NO):
            raise RuntimeError("You are using {cls} as a scheduler but you haven't enabled evaluation during training. This means your trials will not report intermediate results to Ray Tune, and can thus not be stopped early or used to exploit other trials parameters. If this is what you want, do not use {cls}. If you would like to use {cls}, make sure you pass `do_eval=True` and `evaluation_strategy='steps'` in the Trainer `args`.".format(cls=type(kwargs['scheduler']).__name__))
    trainable = ray.tune.with_parameters(_objective, local_trainer=trainer)

    @functools.wraps(trainable)
    def dynamic_modules_import_trainable(*args, **kwargs):
        """
        Wrapper around `tune.with_parameters` to ensure datasets_modules are loaded on each Actor.

        Without this, an ImportError will be thrown. See https://github.com/huggingface/transformers/issues/11565.

        Assumes that `_objective`, defined above, is a function.
        """
        if is_datasets_available():
            import datasets.load
            dynamic_modules_path = os.path.join(datasets.load.init_dynamic_modules(), '__init__.py')
            spec = importlib.util.spec_from_file_location('datasets_modules', dynamic_modules_path)
            datasets_modules = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = datasets_modules
            spec.loader.exec_module(datasets_modules)
        return trainable(*args, **kwargs)
    if hasattr(trainable, '__mixins__'):
        dynamic_modules_import_trainable.__mixins__ = trainable.__mixins__
    analysis = ray.tune.run(dynamic_modules_import_trainable, config=trainer.hp_space(None), num_samples=n_trials, **kwargs)
    best_trial = analysis.get_best_trial(metric='objective', mode=direction[:3], scope=trainer.args.ray_scope)
    best_run = BestRun(best_trial.trial_id, best_trial.last_result['objective'], best_trial.config, analysis)
    if _tb_writer is not None:
        trainer.add_callback(_tb_writer)
    return best_run