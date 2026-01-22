from typing import Any, Callable, Dict, List, Optional
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.v8.classify.train import ClassificationTrainer
import wandb
from wandb.sdk.lib import telemetry
def on_pretrain_routine_end(self, trainer: BaseTrainer) -> None:
    assert self.run is not None
    self.run.summary.update({'model/parameters': get_num_params(trainer.model), 'model/GFLOPs': round(get_flops(trainer.model), 3)})