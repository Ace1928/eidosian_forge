import wandb.data_types as data_types
def metric_is_wandb_dict(metric):
    return '_type' in list(metric.keys()) and metric['_type'] in WANDB_TYPES