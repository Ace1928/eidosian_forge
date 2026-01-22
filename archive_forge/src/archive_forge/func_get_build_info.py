import os.path as _os_path
import platform as _platform
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework.versions import CXX11_ABI_FLAG as _CXX11_ABI_FLAG
from tensorflow.python.framework.versions import CXX_VERSION as _CXX_VERSION
from tensorflow.python.framework.versions import MONOLITHIC_BUILD as _MONOLITHIC_BUILD
from tensorflow.python.framework.versions import VERSION as _VERSION
from tensorflow.python.platform import build_info
from tensorflow.python.util.tf_export import tf_export
@tf_export('sysconfig.get_build_info')
def get_build_info():
    """Get a dictionary describing TensorFlow's build environment.

  Values are generated when TensorFlow is compiled, and are static for each
  TensorFlow package. The return value is a dictionary with string keys such as:

    - cuda_version
    - cudnn_version
    - is_cuda_build
    - is_rocm_build
    - msvcp_dll_names
    - nvcuda_dll_name
    - cudart_dll_name
    - cudnn_dll_name

  Note that the actual keys and values returned by this function is subject to
  change across different versions of TensorFlow or across platforms.

  Returns:
    A Dictionary describing TensorFlow's build environment.
  """
    return build_info.build_info