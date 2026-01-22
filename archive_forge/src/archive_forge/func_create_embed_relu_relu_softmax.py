import random
import pytest
from thinc.api import (
def create_embed_relu_relu_softmax(depth, width, vector_length):
    with Model.define_operators({'>>': chain}):
        model = strings2arrays() >> with_array(HashEmbed(width, vector_length, column=0) >> expand_window(window_size=1) >> Relu(width, width * 3) >> Relu(width, width) >> Softmax(17, width))
    return model