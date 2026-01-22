from functools import wraps
import optree
import optree.utils
from keras.src.backend.common.global_state import get_global_attribute
from keras.src.backend.common.global_state import set_global_attribute
from keras.src.utils import python_utils
def is_in_store(self, store_name, value):
    return id(value) in self.stored_ids[store_name]