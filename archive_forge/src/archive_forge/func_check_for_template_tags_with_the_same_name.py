import copy
from collections import defaultdict
from django.conf import settings
from django.template.backends.django import get_template_tag_modules
from . import Error, Tags, Warning, register
@register(Tags.templates)
def check_for_template_tags_with_the_same_name(app_configs, **kwargs):
    errors = []
    libraries = defaultdict(set)
    for conf in settings.TEMPLATES:
        custom_libraries = conf.get('OPTIONS', {}).get('libraries', {})
        for module_name, module_path in custom_libraries.items():
            libraries[module_name].add(module_path)
    for module_name, module_path in get_template_tag_modules():
        libraries[module_name].add(module_path)
    for library_name, items in libraries.items():
        if len(items) > 1:
            errors.append(Warning(W003.msg.format(repr(library_name), ', '.join((repr(item) for item in sorted(items)))), id=W003.id))
    return errors