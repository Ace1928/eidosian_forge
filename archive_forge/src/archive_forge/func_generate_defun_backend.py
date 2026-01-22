import uuid
import tensorflow.compat.v2 as tf
from tensorflow.python.eager.context import get_device_name
def generate_defun_backend(unique_api_name, preferred_device, func, supportive_attributes):
    function_attributes = {_FUNCTION_API_NAME_ATTRIBUTE: unique_api_name, _FUNCTION_DEVICE_ATTRIBUTE: preferred_device}
    function_attributes.update(supportive_attributes)
    return tf.function(func, autograph=False, experimental_attributes=function_attributes)