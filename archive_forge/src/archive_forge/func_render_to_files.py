import os
import re
import sys
import inspect
import logging
from abc import ABC, ABCMeta
from fnmatch import fnmatch
from pathlib import Path
from typing import Optional, List
from jinja2 import Environment, ChoiceLoader, FileSystemLoader, \
from elementpath import datatypes
import xmlschema
from xmlschema.validators import XsdType, XsdElement, XsdAttribute
from xmlschema.names import XSD_NAMESPACE
def render_to_files(self, names, parent=None, global_vars=None, output_dir='.', force=False):
    if isinstance(names, str):
        names = [names]
    elif not all((isinstance(x, str) for x in names)):
        raise TypeError("'names' argument must contain only strings!")
    template_names = []
    for name in names:
        if is_shell_wildcard(name):
            template_names.extend(self.matching_templates(name))
        else:
            template_names.append(name)
    output_dir = Path(output_dir)
    rendered = []
    for name in template_names:
        try:
            template = self._env.get_template(name, parent, global_vars)
        except TemplateNotFound as err:
            logger.debug('name %r: %s', name, str(err))
        except TemplateAssertionError as err:
            logger.warning('template %r: %s', name, str(err))
        else:
            output_file = output_dir.joinpath(Path(name).name).with_suffix('')
            if not force and output_file.exists():
                continue
            result = template.render(schema=self.schema)
            logger.info('write file %r', str(output_file))
            with open(output_file, 'w') as fp:
                fp.write(result)
            rendered.append(str(output_file))
    return rendered