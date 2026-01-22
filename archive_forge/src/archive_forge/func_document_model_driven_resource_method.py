from botocore.docs.method import document_model_driven_method
def document_model_driven_resource_method(section, method_name, operation_model, event_emitter, method_description=None, example_prefix=None, include_input=None, include_output=None, exclude_input=None, exclude_output=None, document_output=True, resource_action_model=None, include_signature=True):
    document_model_driven_method(section=section, method_name=method_name, operation_model=operation_model, event_emitter=event_emitter, method_description=method_description, example_prefix=example_prefix, include_input=include_input, include_output=include_output, exclude_input=exclude_input, exclude_output=exclude_output, document_output=document_output, include_signature=include_signature)
    if resource_action_model.resource:
        if 'return' in section.available_sections:
            section.delete_section('return')
        resource_type = resource_action_model.resource.type
        new_return_section = section.add_new_section('return')
        return_resource_type = '%s.%s' % (operation_model.service_model.service_name, resource_type)
        return_type = ':py:class:`%s`' % return_resource_type
        return_description = '%s resource' % resource_type
        if _method_returns_resource_list(resource_action_model.resource):
            return_type = 'list(%s)' % return_type
            return_description = 'A list of %s resources' % resource_type
        new_return_section.style.new_line()
        new_return_section.write(':rtype: %s' % return_type)
        new_return_section.style.new_line()
        new_return_section.write(':returns: %s' % return_description)
        new_return_section.style.new_line()