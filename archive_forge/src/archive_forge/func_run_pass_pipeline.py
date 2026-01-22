from tensorflow.python import pywrap_mlir
from tensorflow.python.util.tf_export import tf_export
@tf_export('mlir.experimental.run_pass_pipeline')
def run_pass_pipeline(mlir_txt, pass_pipeline, show_debug_info=False):
    """Runs a pipeline over input module.

  Args:
    mlir_txt: Textual representation of the MLIR module.
    pass_pipeline: Pass pipeline to run on module.
    show_debug_info: Whether to include locations in the emitted textual form.

  Returns:
    A textual representation of the MLIR module corresponding to the
    transformed module.
  """
    return pywrap_mlir.experimental_run_pass_pipeline(mlir_txt, pass_pipeline, show_debug_info)