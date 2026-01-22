import io
import os
from tensorflow.lite.toco.logging import toco_conversion_log_pb2 as _toco_conversion_log_pb2
from tensorflow.python.lib.io import file_io as _file_io
from tensorflow.python.platform import resource_loader as _resource_loader
Generates the HTML report and writes it to local directory.

    This function uses the fields in `toco_conversion_log_before` and
    `toco_conversion_log_after` to populate the HTML content. Certain markers
    (placeholders) in the HTML template are then substituted with the fields
    from the protos. Once finished it will write the HTML file to the specified
    local file path.

    Args:
      toco_conversion_log_before: A `TocoConversionLog` protobuf generated
        before the model is converted by TOCO.
      toco_conversion_log_after: A `TocoConversionLog` protobuf generated after
        the model is converted by TOCO.
      post_training_quant_enabled: A boolean, whether post-training quantization
        is enabled.
      dot_before: A string, the dot representation of the model
        before the conversion.
      dot_after: A string, the dot representation of the model after
        the conversion.
      toco_err_log: A string, the logs emitted by TOCO during conversion. Caller
        need to ensure that this string is properly anonymized (any kind of
        user data should be eliminated).
      tflite_graph_path: A string, the filepath to the converted TFLite model.

    Raises:
      RuntimeError: When error occurs while generating the template.
    