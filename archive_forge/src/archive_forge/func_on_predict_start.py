import copy
from datetime import datetime
from typing import Callable, Dict, Optional, Union
from packaging import version
import wandb
from wandb.sdk.lib import telemetry
def on_predict_start(self, predictor: PREDICTOR_TYPE):
    wandb.run or wandb.init(project=predictor.args.project or 'YOLOv8', config=vars(predictor.args), job_type='prediction_' + predictor.args.task)
    if isinstance(predictor, SAMPredictor):
        self.prompts = copy.deepcopy(predictor.prompts)
        self.prediction_table = wandb.Table(columns=['Image'])