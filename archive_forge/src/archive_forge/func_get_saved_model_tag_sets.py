import os
from google.protobuf import message
from google.protobuf import text_format
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants
from tensorflow.python.util import compat
def get_saved_model_tag_sets(saved_model_dir):
    """Retrieves all the tag-sets available in the SavedModel.

  Args:
    saved_model_dir: Directory containing the SavedModel.

  Returns:
    List of all tag-sets in the SavedModel, where a tag-set is represented as a
    list of strings.
  """
    saved_model = read_saved_model(saved_model_dir)
    all_tags = []
    for meta_graph_def in saved_model.meta_graphs:
        all_tags.append(list(meta_graph_def.meta_info_def.tags))
    return all_tags