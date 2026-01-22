from typing import Optional
import wandb
from wandb import util
from wandb.plots.utils import test_missing, test_types
Calculate and visualize receiver operating characteristic (ROC) scores.

    Arguments:
        y_true (arr): true sparse labels
        y_probas (arr): Target scores, can either be probability estimates, confidence
                         values, or non-thresholded measure of decisions.
                         shape: (*y_true.shape, num_classes)
        labels (list): Named labels for target variable (y). Makes plots easier to
                        read by replacing target values with corresponding index.
                        For example labels = ['dog', 'cat', 'owl'] all 0s are
                        replaced by 'dog', 1s by 'cat'.
        classes_to_plot (list): unique values of y_true to include in the plot
        split_table (bool): If True, adds "Custom Chart Tables/" to the key of the table so that it's logged in a different section.

    Returns:
        Nothing. To see plots, go to your W&B run page then expand the 'media' tab
            under 'auto visualizations'.

    Example:
        ```
        wandb.log({'roc-curve': wandb.plot.roc_curve(y_true, y_probas, labels)})
        ```
    