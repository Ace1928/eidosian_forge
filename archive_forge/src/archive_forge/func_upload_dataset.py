from the local database to W&B in Tables format.
import wandb
from wandb.integration.prodigy import upload_dataset
import base64
import collections.abc
import io
import urllib
from copy import deepcopy
import pandas as pd
from PIL import Image
import wandb
from wandb import util
from wandb.plots.utils import test_missing
from wandb.sdk.lib import telemetry as wb_telemetry
def upload_dataset(dataset_name):
    """Upload dataset from local database to Weights & Biases.

    Args:
        dataset_name: The name of the dataset in the Prodigy database.
    """
    if wandb.run is None:
        raise ValueError('You must call wandb.init() before upload_dataset()')
    with wb_telemetry.context(run=wandb.run) as tel:
        tel.feature.prodigy = True
    prodigy_db = util.get_module('prodigy.components.db', required='`prodigy` library is required but not installed. Please see https://prodi.gy/docs/install')
    database = prodigy_db.connect()
    data = database.get_dataset(dataset_name)
    array_dict_types = []
    schema = get_schema(data, {}, array_dict_types)
    for i, _d in enumerate(data):
        standardize(data[i], schema, array_dict_types)
    table = create_table(data)
    wandb.log({dataset_name: table})
    print('Prodigy dataset `' + dataset_name + '` uploaded.')