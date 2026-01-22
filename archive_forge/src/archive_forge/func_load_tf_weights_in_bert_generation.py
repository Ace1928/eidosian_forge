from ..utils import DummyObject, requires_backends
def load_tf_weights_in_bert_generation(*args, **kwargs):
    requires_backends(load_tf_weights_in_bert_generation, ['torch'])