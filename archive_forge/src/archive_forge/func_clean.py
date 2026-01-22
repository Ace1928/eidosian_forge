import sys
import os
from typing import List, Tuple, Optional
from pathlib import Path
from argparse import ArgumentParser, RawTextHelpFormatter
from project import (QmlProjectData, check_qml_decorators, is_python_file,
def clean(self):
    """Clean build artifacts."""
    for sub_project_file in self.project.sub_projects_files:
        Project(project_file=sub_project_file).clean()
    for file in self.project.files:
        self._clean_file(file)
    if self._qml_module_dir and self._qml_module_dir.is_dir():
        remove_path(self._qml_module_dir)
        if self._qml_module_dir.parent != self.project.project_file.parent:
            project_dir_parts = len(self.project.project_file.parent.parts)
            first_module_dir = self._qml_module_dir.parts[project_dir_parts]
            remove_path(self.project.project_file.parent / first_module_dir)