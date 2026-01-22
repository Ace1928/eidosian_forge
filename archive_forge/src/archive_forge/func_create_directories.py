import os
def create_directories(base_path, tree_structure):
    for path in tree_structure:
        try:
            os.makedirs(os.path.join(base_path, path))
            print(f'Directory created: {path}')
        except FileExistsError:
            print(f'Directory already exists: {path}')