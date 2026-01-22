from typing import Mapping, TypeVar
from .error_reporting import ValidationError
def validate_project_dynamic(pyproject: T) -> T:
    project_table = pyproject.get('project', {})
    dynamic = project_table.get('dynamic', [])
    for field in dynamic:
        if field in project_table:
            msg = f'You cannot provide a value for `project.{field}` and '
            msg += 'list it under `project.dynamic` at the same time'
            name = f'data.project.{field}'
            value = {field: project_table[field], '...': ' # ...', 'dynamic': dynamic}
            raise RedefiningStaticFieldAsDynamic(msg, value, name, rule='PEP 621')
    return pyproject