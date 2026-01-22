import argparse
import os
import gc
import random
import ray
import orjson
import pyarrow
from pyarrow import parquet
def generate_epoch(seed: int, model_type: str, model_path: str, in_filename: str, out_filename: str, per_sequence_loss: bool):
    metadata = {'model_type': model_type}
    schema = [pyarrow.field('total_length', pyarrow.int32()), pyarrow.field('num_seqs', pyarrow.float32()), pyarrow.field(f'seqlens', pyarrow.list_(pyarrow.int32())), pyarrow.field(f'nz_input_ids', pyarrow.list_(pyarrow.int32())), pyarrow.field(f'nz_position_ids', pyarrow.list_(pyarrow.int32())), pyarrow.field(f'nz_shifted_label_ids', pyarrow.list_(pyarrow.int32())), pyarrow.field(f'nz_shifted_loss_weights', pyarrow.list_(pyarrow.float32()))]
    schema = pyarrow.schema(schema, metadata={'metadata_json': orjson.dumps(metadata)})
    with open(in_filename, 'rb') as f:
        batches = f.readlines()
        random.seed(seed)
        random.shuffle(batches)
        batches = _split(batches, int(ray.available_resources()['CPU']))
    handles = [convert_conversation_batch.remote(model_type=model_type, model_path=model_path, batch=batch, schema=schema, per_sequence_loss=per_sequence_loss) for batch in batches]
    parquet.write_table(pyarrow.concat_tables([ray.get(handle) for handle in handles]), out_filename)