import ast
import logging
import os
import re
import sys
import warnings
from typing import List
from importlib import util
from importlib.metadata import version
from pathlib import Path
from . import Nuitka, run_command
def find_pyside_modules(project_dir: Path, extra_ignore_dirs: List[Path]=None, project_data=None):
    """
    Searches all the python files in the project to find all the PySide modules used by
    the application.
    """
    all_modules = set()
    mod_pattern = re.compile('PySide6.Qt(?P<mod_name>.*)')

    def pyside_imports(py_file: Path):
        modules = []
        contents = py_file.read_text(encoding='utf-8')
        try:
            tree = ast.parse(contents)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    main_mod_name = node.module
                    if main_mod_name.startswith('PySide6'):
                        if main_mod_name == 'PySide6':
                            for imported_module in node.names:
                                full_mod_name = imported_module.name
                                if full_mod_name.startswith('Qt'):
                                    modules.append(full_mod_name[2:])
                            continue
                        match = mod_pattern.search(main_mod_name)
                        if match:
                            mod_name = match.group('mod_name')
                            modules.append(mod_name)
                        else:
                            logging.warning(f'[DEPLOY] Unable to find module name from{ast.dump(node)}')
                if isinstance(node, ast.Import):
                    for imported_module in node.names:
                        full_mod_name = imported_module.name
                        if full_mod_name == 'PySide6':
                            logging.warning(IMPORT_WARNING_PYSIDE.format(str(py_file)))
        except Exception as e:
            raise RuntimeError(f'[DEPLOY] Finding module import failed on file {str(py_file)} with error {e}')
        return set(modules)
    py_candidates = []
    ignore_dirs = ['__pycache__', 'env', 'venv', 'deployment']
    if project_data:
        py_candidates = project_data.python_files
        ui_candidates = project_data.ui_files
        qrc_candidates = project_data.qrc_files
        ui_py_candidates = None
        qrc_ui_candidates = None
        if ui_candidates:
            ui_py_candidates = [file.parent / f'ui_{file.stem}.py' for file in ui_candidates if (file.parent / f'ui_{file.stem}.py').exists()]
            if len(ui_py_candidates) != len(ui_candidates):
                warnings.warn("[DEPLOY] The number of uic files and their corresponding Python files don't match.", category=RuntimeWarning)
            py_candidates.extend(ui_py_candidates)
        if qrc_candidates:
            qrc_ui_candidates = [file.parent / f'rc_{file.stem}.py' for file in qrc_candidates if (file.parent / f'rc_{file.stem}.py').exists()]
            if len(qrc_ui_candidates) != len(qrc_candidates):
                warnings.warn("[DEPLOY] The number of qrc files and their corresponding Python files don't match.", category=RuntimeWarning)
            py_candidates.extend(qrc_ui_candidates)
        for py_candidate in py_candidates:
            all_modules = all_modules.union(pyside_imports(py_candidate))
        return list(all_modules)
    if extra_ignore_dirs:
        ignore_dirs.extend(extra_ignore_dirs)
    _walk = os.walk(project_dir)
    for root, dirs, files in _walk:
        dirs[:] = [d for d in dirs if d not in ignore_dirs and (not d.startswith('.'))]
        for py_file in files:
            if py_file.endswith('.py'):
                py_candidates.append(Path(root) / py_file)
    for py_candidate in py_candidates:
        all_modules = all_modules.union(pyside_imports(py_candidate))
    if not all_modules:
        ValueError('[DEPLOY] No PySide6 modules were found')
    return list(all_modules)