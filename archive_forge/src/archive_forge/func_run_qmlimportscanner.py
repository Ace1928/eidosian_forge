import json
import subprocess
import sys
from pathlib import Path
from typing import List
def run_qmlimportscanner(qml_files: List[Path], dry_run: bool):
    """
        Runs pyside6-qmlimportscanner to find all the imported qml modules
    """
    if not qml_files:
        return []
    qml_modules = []
    cmd = ['pyside6-qmlimportscanner', '-qmlFiles']
    cmd.extend([str(qml_file) for qml_file in qml_files])
    if dry_run:
        run_command(command=cmd, dry_run=True)
    _, json_string = run_command(command=cmd, dry_run=False, fetch_output=True)
    json_string = json_string.decode('utf-8')
    json_array = json.loads(json_string)
    qml_modules = [item['name'] for item in json_array if item['type'] == 'module']
    return qml_modules