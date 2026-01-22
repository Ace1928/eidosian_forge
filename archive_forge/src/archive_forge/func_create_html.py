import json
import os
import re
import sys
import numpy as np
def create_html(tflite_input, input_is_filepath=True):
    """Returns html description with the given tflite model.

  Args:
    tflite_input: TFLite flatbuffer model path or model object.
    input_is_filepath: Tells if tflite_input is a model path or a model object.

  Returns:
    Dump of the given tflite model in HTML format.

  Raises:
    RuntimeError: If the input is not valid.
  """
    if input_is_filepath:
        if not os.path.exists(tflite_input):
            raise RuntimeError('Invalid filename %r' % tflite_input)
        if tflite_input.endswith('.tflite') or tflite_input.endswith('.bin'):
            with open(tflite_input, 'rb') as file_handle:
                file_data = bytearray(file_handle.read())
            data = CreateDictFromFlatbuffer(file_data)
        elif tflite_input.endswith('.json'):
            data = json.load(open(tflite_input))
        else:
            raise RuntimeError('Input file was not .tflite or .json')
    else:
        data = CreateDictFromFlatbuffer(tflite_input)
    html = ''
    html += _CSS
    html += '<h1>TensorFlow Lite Model</h2>'
    data['filename'] = tflite_input if input_is_filepath else 'Null (used model object)'
    toplevel_stuff = [('filename', None), ('version', None), ('description', None)]
    html += '<table>\n'
    for key, mapping in toplevel_stuff:
        if not mapping:
            mapping = lambda x: x
        html += '<tr><th>%s</th><td>%s</td></tr>\n' % (key, mapping(data.get(key)))
    html += '</table>\n'
    buffer_keys_to_display = [('data', DataSizeMapper())]
    operator_keys_to_display = [('builtin_code', BuiltinCodeToName), ('custom_code', NameListToString), ('version', None)]
    for d in data['operator_codes']:
        d['builtin_code'] = max(d['builtin_code'], d['deprecated_builtin_code'])
    for subgraph_idx, g in enumerate(data['subgraphs']):
        html += "<div class='subgraph'>"
        tensor_mapper = TensorMapper(g)
        opcode_mapper = OpCodeMapper(data)
        op_keys_to_display = [('inputs', tensor_mapper), ('outputs', tensor_mapper), ('builtin_options', None), ('opcode_index', opcode_mapper)]
        tensor_keys_to_display = [('name', NameListToString), ('type', TensorTypeToName), ('shape', None), ('shape_signature', None), ('buffer', None), ('quantization', None)]
        html += '<h2>Subgraph %d</h2>\n' % subgraph_idx
        html += '<h3>Inputs/Outputs</h3>\n'
        html += GenerateTableHtml([{'inputs': g['inputs'], 'outputs': g['outputs']}], [('inputs', tensor_mapper), ('outputs', tensor_mapper)], display_index=False)
        html += '<h3>Tensors</h3>\n'
        html += GenerateTableHtml(g['tensors'], tensor_keys_to_display)
        if g['operators']:
            html += '<h3>Ops</h3>\n'
            html += GenerateTableHtml(g['operators'], op_keys_to_display)
        html += "<svg id='subgraph%d' width='1600' height='900'></svg>\n" % (subgraph_idx,)
        html += GenerateGraph(subgraph_idx, g, opcode_mapper)
        html += '</div>'
    html += '<h2>Buffers</h2>\n'
    html += GenerateTableHtml(data['buffers'], buffer_keys_to_display)
    html += '<h2>Operator Codes</h2>\n'
    html += GenerateTableHtml(data['operator_codes'], operator_keys_to_display)
    html += '</body></html>\n'
    return html