from typing import Any
import numpy as np
import onnx
from onnx import NodeProto
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_tf_batch_onlybigrams_skip5() -> None:
    input = np.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]]).astype(np.int32)
    output = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]).astype(np.float32)
    ngram_counts = np.array([0, 4]).astype(np.int64)
    ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
    pool_int64s = np.array([2, 3, 5, 4, 5, 6, 7, 8, 6, 7]).astype(np.int64)
    helper = TfIdfVectorizerHelper(mode='TF', min_gram_length=2, max_gram_length=2, max_skip_count=5, ngram_counts=ngram_counts, ngram_indexes=ngram_indexes, pool_int64s=pool_int64s)
    node = helper.make_node_noweights()
    expect(node, inputs=[input], outputs=[output], name='test_tfidfvectorizer_tf_batch_onlybigrams_skip5')