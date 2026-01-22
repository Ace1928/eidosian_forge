import codecs
import types
import six
from apitools.base.protorpclite import messages
from apitools.base.protorpclite import util
@util.positional(1)
def import_descriptor_loader(definition_name, importer=__import__):
    """Find objects by importing modules as needed.

    A definition loader is a function that resolves a definition name to a
    descriptor.

    The import finder resolves definitions to their names by importing modules
    when necessary.

    Args:
      definition_name: Name of definition to find.
      importer: Import function used for importing new modules.

    Returns:
      Appropriate descriptor for any describable type located by name.

    Raises:
      DefinitionNotFoundError when a name does not refer to either a definition
      or a module.
    """
    if definition_name.startswith('.'):
        definition_name = definition_name[1:]
    if not definition_name.startswith('.'):
        leaf = definition_name.split('.')[-1]
        if definition_name:
            try:
                module = importer(definition_name, '', '', [leaf])
            except ImportError:
                pass
            else:
                return describe(module)
    try:
        return describe(messages.find_definition(definition_name, importer=__import__))
    except messages.DefinitionNotFoundError as err:
        split_name = definition_name.rsplit('.', 1)
        if len(split_name) > 1:
            parent, child = split_name
            try:
                parent_definition = import_descriptor_loader(parent, importer=importer)
            except messages.DefinitionNotFoundError:
                pass
            else:
                if isinstance(parent_definition, EnumDescriptor):
                    search_list = parent_definition.values or []
                elif isinstance(parent_definition, MessageDescriptor):
                    search_list = parent_definition.fields or []
                else:
                    search_list = []
                for definition in search_list:
                    if definition.name == child:
                        return definition
        raise err