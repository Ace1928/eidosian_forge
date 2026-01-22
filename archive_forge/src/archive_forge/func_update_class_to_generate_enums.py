def update_class_to_generate_enums(class_to_generate):
    class_to_generate['enums'] = ''
    if class_to_generate.get('is_enum', False):
        enums = ''
        for enum in class_to_generate['enum_values']:
            enums += '    %s = %r\n' % (enum.upper(), enum)
        enums += '\n'
        enums += '    VALID_VALUES = %s\n\n' % _OrderedSet(class_to_generate['enum_values']).set_repr()
        class_to_generate['enums'] = enums