from itertools import product
import math
import pytest
import torch
import transformers
from transformers import (
from tests.helpers import TRUE_FALSE, describe_dtype, id_formatter
@pytest.fixture(scope='session', params=product(models, dtypes))
def model_and_tokenizer(request):
    model, tokenizer = get_model_and_tokenizer(request.param)
    yield (request.param, model, tokenizer)
    del model