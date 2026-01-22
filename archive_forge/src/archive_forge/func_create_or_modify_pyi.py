from __future__ import annotations
import ast
import inspect
from abc import ABCMeta
from functools import wraps
from pathlib import Path
from jinja2 import Template
from gradio.events import EventListener
from gradio.exceptions import ComponentDefinitionError
from gradio.utils import no_raise_exception
def create_or_modify_pyi(component_class: type, class_name: str, events: list[str | EventListener]):
    source_file = Path(inspect.getfile(component_class))
    source_code = source_file.read_text()
    current_impl, lineno = extract_class_source_code(source_code, class_name)
    if not (current_impl and lineno):
        raise ValueError("Couldn't find class source code")
    new_interface = create_pyi(current_impl, events)
    pyi_file = source_file.with_suffix('.pyi')
    if not pyi_file.exists():
        last_empty_line_before_class = -1
        lines = source_code.splitlines()
        for i, line in enumerate(lines):
            if line in ['', ' ']:
                last_empty_line_before_class = i
            if i >= lineno:
                break
        lines = lines[:last_empty_line_before_class] + ['from gradio.events import Dependency'] + lines[last_empty_line_before_class:]
        with no_raise_exception():
            pyi_file.write_text('\n'.join(lines))
    current_interface, _ = extract_class_source_code(pyi_file.read_text(), class_name)
    if not current_interface:
        with no_raise_exception():
            with open(str(pyi_file), mode='a') as f:
                f.write(new_interface)
    else:
        contents = pyi_file.read_text()
        contents = contents.replace(current_interface, new_interface.strip())
        current_contents = pyi_file.read_text()
        if current_contents != contents:
            with no_raise_exception():
                pyi_file.write_text(contents)