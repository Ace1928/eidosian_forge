import os
import pathlib
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union
def get_method_definitions(file_path: Union[str, List[str]], files_to_exclude: Set[str], deprecated_files: Set[str], default_output_type: str, method_to_special_output_type: Dict[str, str], root: str='') -> List[str]:
    """
    #.pyi generation for functional DataPipes Process.

    # 1. Find files that we want to process (exclude the ones who don't)
    # 2. Parse method name and signature
    # 3. Remove first argument after self (unless it is "*datapipes"), default args, and spaces
    """
    if root == '':
        root = str(pathlib.Path(__file__).parent.resolve())
    file_path = [file_path] if isinstance(file_path, str) else file_path
    file_path = [os.path.join(root, path) for path in file_path]
    file_paths = find_file_paths(file_path, files_to_exclude=files_to_exclude.union(deprecated_files))
    methods_and_signatures, methods_and_class_names, methods_w_special_output_types, methods_and_doc_strings = parse_datapipe_files(file_paths)
    for fn_name in method_to_special_output_type:
        if fn_name not in methods_w_special_output_types:
            methods_w_special_output_types.add(fn_name)
    method_definitions = []
    for method_name, arguments in methods_and_signatures.items():
        class_name = methods_and_class_names[method_name]
        if method_name in methods_w_special_output_types:
            output_type = method_to_special_output_type[method_name]
        else:
            output_type = default_output_type
        doc_string = ''.join(methods_and_doc_strings[method_name])
        if doc_string == '':
            doc_string = '    ...\n'
        method_definitions.append(f"# Functional form of '{class_name}'\ndef {method_name}({arguments}) -> {output_type}:\n{doc_string}")
    method_definitions.sort(key=lambda s: s.split('\n')[1])
    return method_definitions