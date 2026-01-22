import copy
import functools
import re
import weakref
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src import initializers
from keras.src.utils import io_utils
from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.utils.warmstart_embedding_matrix')
def warmstart_embedding_matrix(base_vocabulary, new_vocabulary, base_embeddings, new_embeddings_initializer='uniform'):
    """Warm start embedding matrix with changing vocab.

    This util can be used to warmstart the embedding layer matrix when
    vocabulary changes between previously saved checkpoint and model.
    Vocabulary change could mean, the size of the new vocab is different or the
    vocabulary is reshuffled or new vocabulary has been added to old vocabulary.
    If the vocabulary size changes, size of the embedding layer matrix also
    changes. This util remaps the old vocabulary embeddings to the new embedding
    layer matrix.

    Example:
    Here is an example that demonstrates how to use the
    `warmstart_embedding_matrix` util.
    >>> import keras
    >>> vocab_base = tf.convert_to_tensor(["unk", "a", "b", "c"])
    >>> vocab_new = tf.convert_to_tensor(
    ...        ["unk", "unk", "a", "b", "c", "d", "e"])
    >>> vectorized_vocab_base = np.random.rand(vocab_base.shape[0], 3)
    >>> vectorized_vocab_new = np.random.rand(vocab_new.shape[0], 3)
    >>> warmstarted_embedding_matrix = warmstart_embedding_matrix(
    ...       base_vocabulary=vocab_base,
    ...       new_vocabulary=vocab_new,
    ...       base_embeddings=vectorized_vocab_base,
    ...       new_embeddings_initializer=keras.initializers.Constant(
    ...         vectorized_vocab_new))

    Here is an example that demonstrates how to get vocabulary and embedding
    weights from layers, use the `warmstart_embedding_matrix` util to remap the
    layer embeddings and continue with model training.
    ```
    # get old and new vocabulary by using layer.get_vocabulary()
    # for example assume TextVectorization layer is used
    base_vocabulary = old_text_vectorization_layer.get_vocabulary()
    new_vocabulary = new_text_vectorization_layer.get_vocabulary()
    # get previous embedding layer weights
    embedding_weights_base = model.get_layer('embedding').get_weights()[0]
    warmstarted_embedding = keras.utils.warmstart_embedding_matrix(
                                  base_vocabulary,
                                  new_vocabulary,
                                  base_embeddings=embedding_weights_base,
                                  new_embeddings_initializer="uniform")
    updated_embedding_variable = tf.Variable(warmstarted_embedding)

    # update embedding layer weights
    model.layers[1].embeddings = updated_embedding_variable
    model.fit(..)
    # continue with model training

    ```

    Args:
        base_vocabulary: The list of vocabulary terms that
          the preexisting embedding matrix `base_embeddings` represents.
          It can be either a 1D array/tensor or a tuple/list of vocabulary
          terms (strings), or a path to a vocabulary text file. If passing a
           file path, the file should contain one line per term in the
           vocabulary.
        new_vocabulary: The list of vocabulary terms for the new vocabulary
           (same format as above).
        base_embeddings: NumPy array or tensor representing the preexisting
          embedding matrix.
        new_embeddings_initializer: Initializer for embedding vectors for
          previously unseen terms to be added to the new embedding matrix (see
          `keras.initializers`). new_embedding matrix
          needs to be specified with "constant" initializer.
          matrix. None means "uniform". Default value is None.

    Returns:
      tf.tensor of remapped embedding layer matrix

    """
    base_vocabulary = convert_vocab_to_list(base_vocabulary)
    new_vocabulary = convert_vocab_to_list(new_vocabulary)
    new_embeddings_initializer = initializers.get(new_embeddings_initializer)
    new_embedding = new_embeddings_initializer(shape=(len(new_vocabulary), base_embeddings.shape[1]), dtype=base_embeddings.dtype)
    base_vocabulary_dict = dict(zip(base_vocabulary, range(len(base_vocabulary))))
    indices_base_vocabulary = []
    indices_new_vocabulary = []
    for index, key in enumerate(new_vocabulary):
        if key in base_vocabulary_dict:
            indices_base_vocabulary.append(base_vocabulary_dict[key])
            indices_new_vocabulary.append(int(index))
    if indices_base_vocabulary:
        values_to_update = tf.gather(base_embeddings, indices_base_vocabulary)
        new_embedding = tf.tensor_scatter_nd_update(new_embedding, tf.expand_dims(indices_new_vocabulary, axis=1), values_to_update)
    return new_embedding