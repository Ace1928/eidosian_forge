import distutils.spawn
import enum
import hashlib
import os as _os
import platform as _platform
import subprocess as _subprocess
import tempfile as _tempfile
from typing import Optional
import warnings
from tensorflow.compiler.mlir.quantization.stablehlo import quantization_options_pb2 as quant_opts_pb2
from tensorflow.lite.python import lite_constants
from tensorflow.lite.python import util
from tensorflow.lite.python import wrap_toco
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import ConverterError
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.lite.python.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics.wrapper import metrics_wrapper as _metrics_wrapper
from tensorflow.lite.toco import model_flags_pb2 as _model_flags_pb2
from tensorflow.lite.toco import toco_flags_pb2 as _conversion_flags_pb2
from tensorflow.lite.toco import types_pb2 as _types_pb2
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.platform import resource_loader as _resource_loader
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export as _tf_export
def build_conversion_flags(inference_type=dtypes.float32, inference_input_type=None, input_format=lite_constants.TENSORFLOW_GRAPHDEF, output_format=lite_constants.TFLITE, default_ranges_stats=None, drop_control_dependency=True, reorder_across_fake_quant=False, allow_custom_ops=False, post_training_quantize=False, quantize_to_float16=False, dump_graphviz_dir=None, dump_graphviz_video=False, target_ops=None, conversion_summary_dir=None, select_user_tf_ops=None, allow_all_select_tf_ops=False, enable_tflite_resource_variables=True, unfold_batchmatmul=False, legalize_custom_tensor_list_ops=False, lower_tensor_list_ops=True, default_to_single_batch_in_tensor_list_ops=False, accumulation_type=None, allow_bfloat16=False, unfold_large_splat_constant=False, supported_backends=None, disable_per_channel_quantization=False, enable_mlir_dynamic_range_quantizer=False, tf_quantization_mode=None, disable_infer_tensor_range=False, use_fake_quant_num_bits=False, enable_dynamic_update_slice=False, preserve_assert_op=False, guarantee_all_funcs_one_use=False, enable_mlir_variable_quantization=False, disable_fuse_mul_and_fc=False, quantization_options: Optional[quant_opts_pb2.QuantizationOptions]=None, mlir_dump_dir=None, mlir_dump_pass_regex=None, mlir_dump_func_regex=None, mlir_enable_timing=None, mlir_print_ir_before=None, mlir_print_ir_after=None, mlir_print_ir_module_scope=None, mlir_elide_elementsattrs_if_larger=None, use_buffer_offset=False, **_):
    """Builds protocol buffer describing a conversion of a model.

  Typically this is to convert from TensorFlow GraphDef to TFLite, in which
  case the default `input_format` and `output_format` are sufficient.

  Args:
    inference_type: Data type of numeric arrays, excluding the input layer.
      (default tf.float32, must be in {tf.float32, tf.int8, tf.uint8})
    inference_input_type: Data type of the numeric arrays in the input layer. If
      `inference_input_type` is in {tf.int8, tf.uint8}, then
      `quantized_input_stats` must be provided. (default is the value assigned
      to `inference_type`, must be in {tf.float32, tf.int8, tf.uint8})
    input_format: Type of data to read. (default TENSORFLOW_GRAPHDEF, must be in
      {TENSORFLOW_GRAPHDEF})
    output_format: Output file format. (default TFLITE, must be in {TFLITE,
      GRAPHVIZ_DOT})
    default_ranges_stats: Tuple of integers representing (min, max) range values
      for all arrays without a specified range. Intended for experimenting with
      quantization via "dummy quantization". (default None)
    drop_control_dependency: Boolean indicating whether to drop control
      dependencies silently. This is due to TFLite not supporting control
      dependencies. (default True)
    reorder_across_fake_quant: Boolean indicating whether to reorder FakeQuant
      nodes in unexpected locations. Used when the location of the FakeQuant
      nodes is preventing graph transformations necessary to convert the graph.
      Results in a graph that differs from the quantized training graph,
      potentially causing differing arithmetic behavior. (default False)
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When false any unknown operation is an error. When true, custom ops are
      created for any op that is unknown. The developer will need to provide
      these to the TensorFlow Lite runtime with a custom resolver. (default
      False)
    post_training_quantize: Boolean indicating whether to quantize the weights
      of the converted float model. Model size will be reduced and there will be
      latency improvements (at the cost of accuracy). (default False) If
      quantization_options is set, all quantization arg will be ignored.
    quantize_to_float16: Boolean indicating whether to convert float buffers to
      float16. (default False)
    dump_graphviz_dir: Full filepath of folder to dump the graphs at various
      stages of processing GraphViz .dot files. Preferred over
      --output_format=GRAPHVIZ_DOT in order to keep the requirements of the
      output file. (default None)
    dump_graphviz_video: Boolean indicating whether to dump the graph after
      every graph transformation. (default False)
    target_ops: Experimental flag, subject to change. Set of OpsSet options
      indicating which converter to use. (default set([OpsSet.TFLITE_BUILTINS]))
    conversion_summary_dir: A string, the path to the generated conversion logs.
    select_user_tf_ops: List of user's defined TensorFlow ops need to be
      supported in the TensorFlow Lite runtime. These ops will be supported as
      select TensorFlow ops.
    allow_all_select_tf_ops: If True, automatically add all TF ops (including
      custom TF ops) to the converted model as flex ops.
    enable_tflite_resource_variables: Experimental flag, subject to change.
      Enables conversion of resource variables. (default False)
    unfold_batchmatmul: Whether to unfold tf.BatchMatMul to a set of
      tfl.fully_connected ops. If not, translate to tfl.batch_matmul.
    legalize_custom_tensor_list_ops: Whether to legalize `tf.TensorList*` ops to
      tfl custom if they can all be supported.
    lower_tensor_list_ops: Whether to lower tensor list ops to builtin ops. If
      not, use Flex tensor list ops.
    default_to_single_batch_in_tensor_list_ops: Whether to force to use batch
      size one when the tensor list ops has the unspecified batch size.
    accumulation_type: Data type of the accumulators in quantized inference.
      Typically used for float16 quantization and is either fp16 or fp32.
    allow_bfloat16: Whether the converted model supports reduced precision
      inference with the bfloat16 type.
    unfold_large_splat_constant: Whether to unfold large splat constant tensors
      in the flatbuffer model to reduce size.
    supported_backends: List of TFLite backends which needs to check
      compatibility.
    disable_per_channel_quantization: Disable per-channel quantized weights for
      dynamic range quantization. Only per-tensor quantization will be used.
    enable_mlir_dynamic_range_quantizer: Enable MLIR dynamic range quantization.
      If False, the old converter dynamic range quantizer is used.
    tf_quantization_mode: Indicates the mode of TF Quantization when the output
      model is used for TF Quantization.
    disable_infer_tensor_range: Disable infering tensor ranges.
    use_fake_quant_num_bits: Allow quantization parameters to be calculated from
      num_bits attribute.
    enable_dynamic_update_slice: Enable to convert to DynamicUpdateSlice op.
      (default: False).
    preserve_assert_op: Whether to preserve `TF::AssertOp` (default: False).
    guarantee_all_funcs_one_use: Whether to clone functions so that each
      function only has a single use. This option will be helpful if the
      conversion fails when the `PartitionedCall` or `StatefulPartitionedCall`
      can't be properly inlined (default: False).
    enable_mlir_variable_quantization: Enable MLIR variable quantization. There
      is a variable freezing pass, but some variables may not be fully frozen by
      it. This flag enables quantization of those residual variables in the MLIR
      graph.
    disable_fuse_mul_and_fc: Disable fusing input multiplication with
      fullyconnected operations. Useful when quantizing weights.
    quantization_options: Config to indicate quantization options of each
      components (ex: weight, bias, activation). This can be a preset method or
      a custom method, and allows finer, modular control. This option will
      override any other existing quantization flags. We plan on gradually
      migrating all quantization-related specs into this option.
    mlir_dump_dir: A string specifying the target directory to output MLIR dumps
      produced during conversion. If populated, enables MLIR dumps.
    mlir_dump_pass_regex: A string containing a regular expression for filtering
      the pass names to be dumped. Effective only if `mlir_dump_dir` is
      populated.
    mlir_dump_func_regex: A string containing a regular expression for filtering
      the function names to be dumped. Effective only if `mlir_dump_dir` is
      populated.
    mlir_enable_timing: A boolean, if set to true reports the execution time of
      each MLIR pass.
    mlir_print_ir_before: A string containing a regular expression. If
      specified, prints MLIR before passes which match.
    mlir_print_ir_after: A string containing a regular expression. If specified,
      prints MLIR after passes which match.
    mlir_print_ir_module_scope: A boolean, if set to true always print the
      top-level operation when printing IR for print_ir_[before|after].
    mlir_elide_elementsattrs_if_larger: An int, if specified elides
      ElementsAttrs with '...' that have more elements than the given upper
      limit.
    use_buffer_offset: Force the model use buffer_offset & buffer_size fields
      instead of data. i.e. store the constant tensor and custom op binaries
      outside of Flatbuffers

  Returns:
    conversion_flags: protocol buffer describing the conversion process.
  Raises:
    ValueError, if the input tensor type is unknown.
  """
    conversion_flags = _conversion_flags_pb2.TocoFlags()
    conversion_flags.inference_type = convert_inference_tf_type_to_tflite_type(inference_type, usage='inference_type flag')
    if inference_input_type:
        conversion_flags.inference_input_type = convert_inference_tf_type_to_tflite_type(inference_input_type, usage='inference_input_type flag')
    else:
        conversion_flags.inference_input_type = conversion_flags.inference_type
    conversion_flags.input_format = input_format
    conversion_flags.output_format = output_format
    if default_ranges_stats:
        conversion_flags.default_ranges_min = default_ranges_stats[0]
        conversion_flags.default_ranges_max = default_ranges_stats[1]
    conversion_flags.drop_control_dependency = drop_control_dependency
    conversion_flags.reorder_across_fake_quant = reorder_across_fake_quant
    conversion_flags.allow_custom_ops = allow_custom_ops
    conversion_flags.post_training_quantize = post_training_quantize
    conversion_flags.quantize_to_float16 = quantize_to_float16
    if dump_graphviz_dir:
        conversion_flags.dump_graphviz_dir = dump_graphviz_dir
    conversion_flags.dump_graphviz_include_video = dump_graphviz_video
    if target_ops:
        if OpsSet.SELECT_TF_OPS in target_ops:
            conversion_flags.enable_select_tf_ops = True
        if set(target_ops) == {OpsSet.SELECT_TF_OPS}:
            conversion_flags.force_select_tf_ops = True
        if OpsSet.EXPERIMENTAL_STABLEHLO_OPS in target_ops:
            conversion_flags.convert_to_stablehlo = True
        if OpsSet.EXPERIMENTAL_STABLEHLO_OPS in target_ops and len(target_ops) > 1:
            raise ValueError('StableHLO Ops set can not be specified with other Ops set together')
    if conversion_summary_dir:
        conversion_flags.conversion_summary_dir = conversion_summary_dir
    if select_user_tf_ops:
        conversion_flags.select_user_tf_ops.extend(select_user_tf_ops)
    conversion_flags.allow_all_select_tf_ops = allow_all_select_tf_ops
    conversion_flags.enable_tflite_resource_variables = enable_tflite_resource_variables
    conversion_flags.unfold_batchmatmul = unfold_batchmatmul
    conversion_flags.legalize_custom_tensor_list_ops = legalize_custom_tensor_list_ops
    conversion_flags.lower_tensor_list_ops = lower_tensor_list_ops
    conversion_flags.default_to_single_batch_in_tensor_list_ops = default_to_single_batch_in_tensor_list_ops
    if accumulation_type:
        conversion_flags.accumulation_type = convert_tensor_tf_type_to_tflite_type(accumulation_type, usage='accumulation_type flag')
    conversion_flags.allow_bfloat16 = allow_bfloat16
    conversion_flags.unfold_large_splat_constant = unfold_large_splat_constant
    if supported_backends:
        conversion_flags.supported_backends.extend(supported_backends)
    conversion_flags.disable_per_channel_quantization = disable_per_channel_quantization
    conversion_flags.enable_mlir_dynamic_range_quantizer = enable_mlir_dynamic_range_quantizer
    conversion_flags.enable_dynamic_update_slice = enable_dynamic_update_slice
    conversion_flags.preserve_assert_op = preserve_assert_op
    conversion_flags.guarantee_all_funcs_one_use = guarantee_all_funcs_one_use
    if tf_quantization_mode:
        conversion_flags.tf_quantization_mode = tf_quantization_mode
    conversion_flags.disable_infer_tensor_range = disable_infer_tensor_range
    conversion_flags.use_fake_quant_num_bits = use_fake_quant_num_bits
    conversion_flags.enable_mlir_variable_quantization = enable_mlir_variable_quantization
    conversion_flags.disable_fuse_mul_and_fc = disable_fuse_mul_and_fc
    if quantization_options:
        conversion_flags.quantization_options.CopyFrom(quantization_options)
    if mlir_dump_dir is not None:
        conversion_flags.debug_options.mlir_dump_dir = mlir_dump_dir
    if mlir_dump_pass_regex is not None:
        conversion_flags.debug_options.mlir_dump_pass_regex = mlir_dump_pass_regex
    if mlir_dump_func_regex is not None:
        conversion_flags.debug_options.mlir_dump_func_regex = mlir_dump_func_regex
    if mlir_enable_timing is not None:
        conversion_flags.debug_options.mlir_enable_timing = mlir_enable_timing
    if mlir_print_ir_before is not None:
        conversion_flags.debug_options.mlir_print_ir_before = mlir_print_ir_before
    if mlir_print_ir_after is not None:
        conversion_flags.debug_options.mlir_print_ir_after = mlir_print_ir_after
    if mlir_print_ir_module_scope is not None:
        conversion_flags.debug_options.mlir_print_ir_module_scope = mlir_print_ir_module_scope
    if mlir_elide_elementsattrs_if_larger is not None:
        conversion_flags.debug_options.mlir_elide_elementsattrs_if_larger = mlir_elide_elementsattrs_if_larger
    if use_buffer_offset is not None:
        conversion_flags.use_buffer_offset = use_buffer_offset
    return conversion_flags