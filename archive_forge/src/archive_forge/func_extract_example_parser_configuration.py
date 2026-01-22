from tensorflow.core.example import example_parser_configuration_pb2
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
def extract_example_parser_configuration(parse_example_op, sess):
    """Returns an ExampleParserConfig proto.

  Args:
    parse_example_op: A ParseExample or ParseExampleV2 `Operation`
    sess: A tf.compat.v1.Session needed to obtain some configuration values.
  Returns:
    A ExampleParserConfig proto.

  Raises:
    ValueError: If attributes are inconsistent.
  """
    if parse_example_op.type == 'ParseExample':
        return _extract_from_parse_example(parse_example_op, sess)
    elif parse_example_op.type == 'ParseExampleV2':
        return _extract_from_parse_example_v2(parse_example_op, sess)
    else:
        raise ValueError(f'Found unexpected type when parsing example. Expected `ParseExample` object. Received type: {parse_example_op.type}')