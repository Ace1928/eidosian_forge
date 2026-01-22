import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
import wandb
def log_predictions(self, predictions: Union[Sequence, Dict[str, Sequence]], prediction_col_name: str='output', val_ndx_col_name: str='val_row', table_name: str='validation_predictions', commit: bool=True) -> wandb.data_types.Table:
    """Log a set of predictions.

        Intended usage:

        vl.log_predictions(vl.make_predictions(self.model.predict))

        Args:
            predictions (Sequence | Dict[str, Sequence]): A list of prediction vectors or dictionary
                of lists of prediction vectors
            prediction_col_name (str, optional): the name of the prediction column. Defaults to "output".
            val_ndx_col_name (str, optional): The name of the column linking prediction table
                to the validation ata table. Defaults to "val_row".
            table_name (str, optional): name of the prediction table. Defaults to "validation_predictions".
            commit (bool, optional): determines if commit should be called on the logged data. Defaults to False.
        """
    pred_table = wandb.Table(columns=[], data=[])
    if isinstance(predictions, dict):
        for col_name in predictions:
            pred_table.add_column(col_name, predictions[col_name])
    else:
        pred_table.add_column(prediction_col_name, predictions)
    pred_table.add_column(val_ndx_col_name, self.validation_indexes)
    if self.prediction_row_processor is None and self.infer_missing_processors:
        example_prediction = _make_example(predictions)
        example_input = _make_example(self.validation_inputs)
        if example_prediction is not None and example_input is not None:
            self.prediction_row_processor = _infer_prediction_row_processor(example_prediction, example_input, self.class_labels_table, self.input_col_name, prediction_col_name)
    if self.prediction_row_processor is not None:
        pred_table.add_computed_columns(self.prediction_row_processor)
    wandb.log({table_name: pred_table}, commit=commit)
    return pred_table