def update_class_to_generate_register_dec(classes_to_generate, class_to_generate):
    class_to_generate['register_request'] = ''
    class_to_generate['register_dec'] = '@register'
    properties = class_to_generate.get('properties')
    enum_type = properties.get('type', {}).get('enum')
    command = None
    event = None
    if enum_type and len(enum_type) == 1 and (next(iter(enum_type)) in ('request', 'response', 'event')):
        msg_type = next(iter(enum_type))
        if msg_type == 'response':
            response_name = class_to_generate['name']
            request_name = response_name[:-len('Response')] + 'Request'
            if request_name in classes_to_generate:
                command = classes_to_generate[request_name]['properties'].get('command')
            elif response_name == 'ErrorResponse':
                command = {'enum': ['error']}
            else:
                raise AssertionError('Unhandled: %s' % (response_name,))
        elif msg_type == 'request':
            command = properties.get('command')
        elif msg_type == 'event':
            command = properties.get('event')
        else:
            raise AssertionError('Unexpected condition.')
        if command:
            enum = command.get('enum')
            if enum and len(enum) == 1:
                class_to_generate['register_request'] = '@register_%s(%r)\n' % (msg_type, enum[0])