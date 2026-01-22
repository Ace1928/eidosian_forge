import numpy as np
from tensorboard.compat import tf2 as tf
from tensorboard.compat.proto import summary_pb2
from tensorboard.plugins.text import metadata
from tensorboard.util import tensor_util
def text_pb(tag, data, description=None):
    """Create a text tf.Summary protobuf.

    Arguments:
      tag: String tag for the summary.
      data: A Python bytestring (of type bytes), a Unicode string, or a numpy data
        array of those types.
      description: Optional long-form description for this summary, as a `str`.
        Markdown is supported. Defaults to empty.

    Raises:
      TypeError: If the type of the data is unsupported.

    Returns:
      A `tf.Summary` protobuf object.
    """
    try:
        tensor = tensor_util.make_tensor_proto(data, dtype=np.object_)
    except TypeError as e:
        raise TypeError('tensor must be of type string', e)
    summary_metadata = metadata.create_summary_metadata(display_name=None, description=description)
    summary = summary_pb2.Summary()
    summary.value.add(tag=tag, metadata=summary_metadata, tensor=tensor)
    return summary