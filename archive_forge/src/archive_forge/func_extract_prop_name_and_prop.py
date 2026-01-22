def extract_prop_name_and_prop(class_to_generate):
    properties = class_to_generate.get('properties')
    required = _OrderedSet(class_to_generate.get('required', _OrderedSet()))
    prop_name_and_prop = list(properties.items())

    def compute_sort_key(x):
        key = x[0]
        if key in required:
            if key == 'seq':
                return 0.5
            return 0
        return 1
    prop_name_and_prop.sort(key=compute_sort_key)
    return prop_name_and_prop