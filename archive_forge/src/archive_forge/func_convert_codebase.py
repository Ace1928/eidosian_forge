import os
import shutil
def convert_codebase(package, code_directory='src'):
    if not os.path.exists(package):
        raise ValueError(f"No directory named '{package}'.")
    os.rename(package, code_directory)
    os.mkdir(package)
    shutil.move(code_directory, os.path.join(package, code_directory))
    rewrite_python_file_imports(target_dir=package, root_name=package, offset_name='src', verbose=True)
    with open(os.path.join(package, '__init__.py'), 'w'):
        pass