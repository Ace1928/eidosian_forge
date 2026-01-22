import copy
from datetime import datetime
from typing import Callable, Dict, Optional, Union
from packaging import version
import wandb
from wandb.sdk.lib import telemetry
@torch.no_grad()
def on_val_end(self, trainer: VALIDATOR_TYPE):
    if self.task in self.supported_tasks:
        validator = trainer
        dataloader = validator.dataloader
        class_label_map = validator.names
        if self.task == 'pose':
            self.validation_table = plot_pose_validation_results(dataloader=dataloader, class_label_map=class_label_map, model_name=self.model_name, predictor=self.predictor, visualize_skeleton=self.visualize_skeleton, table=self.validation_table, max_validation_batches=self.max_validation_batches)
        elif self.task == 'segment':
            self.validation_table = plot_segmentation_validation_results(dataloader=dataloader, class_label_map=class_label_map, model_name=self.model_name, predictor=self.predictor, table=self.validation_table, max_validation_batches=self.max_validation_batches)
        elif self.task == 'detect':
            self.validation_table = plot_detection_validation_results(dataloader=dataloader, class_label_map=class_label_map, model_name=self.model_name, predictor=self.predictor, table=self.validation_table, max_validation_batches=self.max_validation_batches)
        elif self.task == 'classify':
            self.validation_table = plot_classification_validation_results(dataloader=dataloader, model_name=self.model_name, predictor=self.predictor, table=self.validation_table, max_validation_batches=self.max_validation_batches)
        wandb.log({'Validation-Table': self.validation_table}, commit=False)