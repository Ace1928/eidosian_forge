import argparse
import os
import gc
import random
import ray
import orjson
import pyarrow
from pyarrow import parquet
def generate_dataset(model_type, model_path, in_prefix, out_prefix, per_sequence_loss, seed):
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=os.cpu_count())
    epoch = 0
    while True:
        in_filename = f'{in_prefix}.{epoch}.jsonl'
        if not os.path.exists(in_filename):
            break
        out_filename = f'{out_prefix}.{epoch}.parquet'
        generate_epoch(seed=seed + epoch, model_type=model_type, model_path=model_path, in_filename=in_filename, out_filename=out_filename, per_sequence_loss=per_sequence_loss)
        gc.collect()
        epoch += 1