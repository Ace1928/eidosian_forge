from tensorflow.python import pywrap_mlir
from tensorflow.python.util.tf_export import tf_export
@tf_export('mlir.experimental.convert_graph_def')
def convert_graph_def(graph_def, pass_pipeline='tf-standard-pipeline', show_debug_info=False):
    """Import a GraphDef and convert it to a textual MLIR module.

  This API is only intended for inspecting the internals of TensorFlow and the
  string returned is at the moment intended for debugging purposes.

  Args:
    graph_def: An object of type graph_pb2.GraphDef or a textual proto
      representation of a valid GraphDef.
    pass_pipeline: A textual description of an MLIR Pass Pipeline to run on the
      module, see MLIR documentation for the [textual pass pipeline
      syntax](https://mlir.llvm.org/docs/PassManagement/#textual-pass-pipeline-specification).
    show_debug_info: Whether to include locations in the emitted textual form.

  Returns:
    A textual representation of the MLIR module corresponding to the graphdef.

  Raises:
    InvalidArgumentError: if graph_def is invalid or cannot be converted to
      MLIR.
  """
    return pywrap_mlir.import_graphdef(graph_def, pass_pipeline, show_debug_info)