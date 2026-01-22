from typing import Any, Callable, Dict, List, Optional
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.yolo.v8.classify.train import ClassificationTrainer
import wandb
from wandb.sdk.lib import telemetry
Property contains all the relevant callbacks to add to the YOLO model for the Weights & Biases logging.