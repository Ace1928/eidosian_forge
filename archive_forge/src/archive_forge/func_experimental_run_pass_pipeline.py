from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python._pywrap_mlir import *
def experimental_run_pass_pipeline(mlir_txt, pass_pipeline, show_debug_info):
    return ExperimentalRunPassPipeline(mlir_txt.encode('utf-8'), pass_pipeline.encode('utf-8'), show_debug_info)