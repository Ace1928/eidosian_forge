from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import constants
from tensorflow.python.util import compat
def get_assets_dir(export_dir):
    """Return path to asset directory in the SavedModel."""
    return file_io.join(compat.as_text(export_dir), compat.as_text(constants.ASSETS_DIRECTORY))