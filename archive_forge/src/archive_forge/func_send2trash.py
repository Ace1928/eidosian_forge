from __future__ import unicode_literals
from ctypes import cdll, byref, Structure, c_char, c_char_p
from ctypes.util import find_library
from send2trash.compat import binary_type
from send2trash.util import preprocess_paths
def send2trash(paths):
    paths = preprocess_paths(paths)
    paths = [path.encode('utf-8') if not isinstance(path, binary_type) else path for path in paths]
    for path in paths:
        fp = FSRef()
        opts = kFSPathMakeRefDoNotFollowLeafSymlink
        op_result = FSPathMakeRefWithOptions(path, opts, byref(fp), None)
        check_op_result(op_result)
        opts = kFSFileOperationDefaultOptions
        op_result = FSMoveObjectToTrashSync(byref(fp), None, opts)
        check_op_result(op_result)