import wandb
from wandb import util
from wandb.plots.utils import (
def pr_table(pr_curves):
    data = []
    count = 0
    for i, class_name in enumerate(pr_curves.keys()):
        precision, recall = pr_curves[class_name]
        for p, r in zip(precision, recall):
            if labels is not None and (isinstance(class_name, int) or isinstance(class_name, np.integer)):
                class_name = labels[class_name]
            data.append([class_name, round(p, 3), round(r, 3)])
            count += 1
            if count >= chart_limit:
                wandb.termwarn('wandb uses only the first %d datapoints to create the plots.' % wandb.Table.MAX_ROWS)
                break
    return wandb.visualize('wandb/pr_curve/v1', wandb.Table(columns=['class', 'precision', 'recall'], data=data))