import functools
import os
from google.protobuf.any_pb2 import Any
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model.pywrap_saved_model import constants
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export
def copy_assets_to_destination_dir(asset_filename_map, destination_dir, saved_files=None):
    """Copy all assets from source path to destination path.

  Args:
    asset_filename_map: a dict of filenames used for saving the asset in
      the SavedModel to full paths from which the filenames were derived.
    destination_dir: the destination directory that assets are stored in.
    saved_files: a set of destination filepaths that have already been copied
      and will be skipped
  """
    if saved_files is None:
        saved_files = set()
    assets_destination_dir = path_helpers.get_or_create_assets_dir(destination_dir)
    for asset_basename, asset_source_filepath in asset_filename_map.items():
        asset_destination_filepath = file_io.join(compat.as_bytes(assets_destination_dir), compat.as_bytes(asset_basename))
        if file_io.file_exists(asset_source_filepath) and asset_source_filepath != asset_destination_filepath and (asset_destination_filepath not in saved_files):
            file_io.copy(asset_source_filepath, asset_destination_filepath, overwrite=True)
            saved_files.add(asset_destination_filepath)
    tf_logging.info('Assets written to: %s', compat.as_text(assets_destination_dir))