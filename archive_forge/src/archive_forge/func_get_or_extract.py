import contextlib
import multiprocessing
import os
import shutil
import tempfile
import zipfile
@staticmethod
def get_or_extract(archive_path):
    """Given a path returned by add_local_model(), this method will return a tuple of
        (loaded_model, local_model_path).
        If this Python process ever loaded the model before, we will reuse that copy.
        """
    from pyspark.files import SparkFiles
    if archive_path in _SparkDirectoryDistributor._extracted_dir_paths:
        return _SparkDirectoryDistributor._extracted_dir_paths[archive_path]
    if archive_path.startswith(_NFS_PATH_PREFIX):
        local_path = archive_path[len(_NFS_PATH_PREFIX):]
    else:
        archive_path_basename = os.path.basename(archive_path)
        local_path = SparkFiles.get(archive_path_basename)
    temp_dir = tempfile.mkdtemp()
    zip_ref = zipfile.ZipFile(local_path, 'r')
    zip_ref.extractall(temp_dir)
    zip_ref.close()
    _SparkDirectoryDistributor._extracted_dir_paths[archive_path] = temp_dir
    return _SparkDirectoryDistributor._extracted_dir_paths[archive_path]