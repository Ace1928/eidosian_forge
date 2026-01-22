import os
def get_cmake_dir() -> str:
    """
    Return the path to the pybind11 CMake module directory.
    """
    cmake_installed_path = os.path.join(DIR, 'share', 'cmake', 'pybind11')
    if os.path.exists(cmake_installed_path):
        return cmake_installed_path
    msg = 'pybind11 not installed, installation required to access the CMake files'
    raise ImportError(msg)