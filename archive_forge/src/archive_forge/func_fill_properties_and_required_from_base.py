def fill_properties_and_required_from_base(classes_to_generate):
    for class_to_generate in classes_to_generate.values():
        dct = {}
        s = _OrderedSet()
        for base_definition in reversed(collect_bases(class_to_generate, classes_to_generate)):
            dct.update(classes_to_generate[base_definition].get('properties', {}))
            s.update(classes_to_generate[base_definition].get('required', _OrderedSet()))
        dct.update(class_to_generate['properties'])
        class_to_generate['properties'] = dct
        s.update(class_to_generate['required'])
        class_to_generate['required'] = s
    return class_to_generate