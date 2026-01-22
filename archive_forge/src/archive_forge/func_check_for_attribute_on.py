from warnings import simplefilter
import numpy as np
import wandb
def check_for_attribute_on(model, attributes_to_check):
    for attr in attributes_to_check:
        if hasattr(model, attr):
            return attr
    return None