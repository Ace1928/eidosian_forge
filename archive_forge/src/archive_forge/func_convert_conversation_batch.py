import argparse
import os
import gc
import random
import ray
import orjson
import pyarrow
from pyarrow import parquet
@ray.remote
def convert_conversation_batch(model_type: str, model_path: str, batch: list, schema: pyarrow.Schema, per_sequence_loss: bool):
    from ochat.config import MODEL_CONFIG_MAP, Conversation
    model_config = MODEL_CONFIG_MAP[model_type]
    tokenizer = model_config.model_tokenizer_create(model_path)
    conv_template = model_config.conversation_template(tokenizer=tokenizer)
    print('Decoding JSON ...')
    batch = [Conversation(**orjson.loads(json_line)) for json_line in batch]
    print('Tokenizing ...')
    tokens_list, weights_list = conv_template.tokenize_conversations(batch, inference=False, seq_level_weight=per_sequence_loss)
    del batch
    gc.collect()
    print('Generating ...')
    max_context = model_config.model_max_context
    outputs = {k: [] for k in schema.names}
    for tokens, weights in zip(tokens_list, weights_list):
        assert len(tokens) == len(weights)
        tokens = tokens[:max_context]
        weights = weights[:max_context]
        add_single_conv(outputs, tokens, weights)
    del tokens_list, weights_list
    gc.collect()
    print('To table ...')
    table = pyarrow.Table.from_pydict(outputs, schema=schema)
    del outputs
    gc.collect()
    print('Chunk finish')
    return table