import wandb
from wandb import util
from wandb.plots.utils import (
def roc_table(fpr_dict, tpr_dict, classes, indices_to_plot):
    data = []
    count = 0
    for i, to_plot in enumerate(indices_to_plot):
        fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true, probas[:, i], pos_label=classes[i])
        if to_plot:
            roc_auc = auc(fpr_dict[i], tpr_dict[i])
            for j in range(len(fpr_dict[i])):
                if labels is not None and (isinstance(classes[i], int) or isinstance(classes[0], np.integer)):
                    class_dict = labels[classes[i]]
                else:
                    class_dict = classes[i]
                fpr = [class_dict, round(fpr_dict[i][j], 3), round(tpr_dict[i][j], 3)]
                data.append(fpr)
                count += 1
                if count >= chart_limit:
                    wandb.termwarn('wandb uses only the first %d datapoints to create the plots.' % wandb.Table.MAX_ROWS)
                    break
    return wandb.visualize('wandb/roc/v1', wandb.Table(columns=['class', 'fpr', 'tpr'], data=data))