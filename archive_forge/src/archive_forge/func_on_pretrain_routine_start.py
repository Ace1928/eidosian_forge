from typing import Any, Callable, Dict, List, Optional
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.v8.classify.train import ClassificationTrainer
import wandb
from wandb.sdk.lib import telemetry
def on_pretrain_routine_start(self, trainer: BaseTrainer) -> None:
    """Starts a new wandb run to track the training process and log to Weights & Biases.

        Args:
            trainer: A task trainer that's inherited from `:class:ultralytics.yolo.engine.trainer.BaseTrainer`
                    that contains the model training and optimization routine.
        """
    if wandb.run is None:
        self.run = wandb.init(name=self.run_name if self.run_name else trainer.args.name, project=self.project if self.project else trainer.args.project or 'YOLOv8', tags=self.tags if self.tags else ['YOLOv8'], config=vars(trainer.args), resume=self.resume if self.resume else None, **self.kwargs)
    else:
        self.run = wandb.run
    assert self.run is not None
    self.run.define_metric('epoch', hidden=True)
    self.run.define_metric('train/*', step_metric='epoch', step_sync=True, summary='min')
    self.run.define_metric('val/*', step_metric='epoch', step_sync=True, summary='min')
    self.run.define_metric('metrics/*', step_metric='epoch', step_sync=True, summary='max')
    self.run.define_metric('lr/*', step_metric='epoch', step_sync=True, summary='last')
    with telemetry.context(run=wandb.run) as tel:
        tel.feature.ultralytics_yolov8 = True