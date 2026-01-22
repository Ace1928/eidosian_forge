from tensorflow.lite.python import wrap_toco
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics._pywrap_tensorflow_lite_metrics_wrapper import MetricsWrapper  # pylint: disable=unused-import
def retrieve_collected_errors():
    """Returns and clears the list of collected errors in ErrorCollector.

  The RetrieveCollectedErrors function in C++ returns a list of serialized proto
  messages. This function will convert them to ConverterErrorData instances.

  Returns:
    A list of ConverterErrorData.
  """
    serialized_message_list = wrap_toco.wrapped_retrieve_collected_errors()
    return list(map(converter_error_data_pb2.ConverterErrorData.FromString, serialized_message_list))