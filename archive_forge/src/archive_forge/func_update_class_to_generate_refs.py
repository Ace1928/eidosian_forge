def update_class_to_generate_refs(class_to_generate):
    properties = class_to_generate['properties']
    class_to_generate['refs'] = '    __refs__ = %s' % _OrderedSet((key for key, val in properties.items() if val['type'].__class__ == Ref)).set_repr()