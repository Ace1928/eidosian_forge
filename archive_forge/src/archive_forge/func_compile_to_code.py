from functools import partial, update_wrapper
from .draft04 import CodeGeneratorDraft04
from .draft06 import CodeGeneratorDraft06
from .draft07 import CodeGeneratorDraft07
from .exceptions import JsonSchemaException, JsonSchemaValueException, JsonSchemaDefinitionException
from .ref_resolver import RefResolver
from .version import VERSION
def compile_to_code(definition, handlers={}, formats={}, use_default=True, use_formats=True):
    """
    Generates validation code for validating JSON schema passed in ``definition``.
    Example:

    .. code-block:: python

        import fastjsonschema

        code = fastjsonschema.compile_to_code({'type': 'string'})
        with open('your_file.py', 'w') as f:
            f.write(code)

    You can also use it as a script:

    .. code-block:: bash

        echo "{'type': 'string'}" | python3 -m fastjsonschema > your_file.py
        python3 -m fastjsonschema "{'type': 'string'}" > your_file.py

    Exception :any:`JsonSchemaDefinitionException` is raised when generating the
    code fails (bad definition).
    """
    _, code_generator = _factory(definition, handlers, formats, use_default, use_formats)
    return 'VERSION = "' + VERSION + '"\n' + code_generator.global_state_code + '\n' + code_generator.func_code