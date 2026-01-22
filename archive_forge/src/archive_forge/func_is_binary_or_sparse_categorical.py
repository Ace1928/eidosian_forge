import tree
from keras.src import backend
from keras.src import losses as losses_module
from keras.src import metrics as metrics_module
from keras.src import ops
from keras.src.utils.naming import get_object_name
def is_binary_or_sparse_categorical(y_true, y_pred):
    y_t_rank = len(y_true.shape)
    y_p_rank = len(y_pred.shape)
    y_t_last_dim = y_true.shape[-1]
    y_p_last_dim = y_pred.shape[-1]
    is_binary = y_p_last_dim == 1
    is_sparse_categorical = y_t_rank < y_p_rank or (y_t_last_dim == 1 and y_p_last_dim > 1)
    return (is_binary, is_sparse_categorical)