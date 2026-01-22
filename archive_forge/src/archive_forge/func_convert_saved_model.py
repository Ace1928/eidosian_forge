from tensorflow.python import pywrap_mlir
from tensorflow.python.util.tf_export import tf_export
@tf_export('mlir.experimental.convert_saved_model')
def convert_saved_model(saved_model_path, exported_names, show_debug_info=False):
    """Converts a SavedModel to MLIR module.

  Args:
    saved_model_path: Path to SavedModel.
    exported_names: Names to export.
    show_debug_info: Whether to include locations in the emitted textual form.

  Returns:
    A textual representation of the MLIR module corresponding to the
    SavedModel.
  """
    return pywrap_mlir.experimental_convert_saved_model_to_mlir(saved_model_path, exported_names, show_debug_info)