def update_class_to_generate_init(class_to_generate):
    args = []
    init_body = []
    docstring = []
    required = _OrderedSet(class_to_generate.get('required', _OrderedSet()))
    prop_name_and_prop = extract_prop_name_and_prop(class_to_generate)
    translate_prop_names = []
    for prop_name, prop in prop_name_and_prop:
        if is_variable_to_translate(class_to_generate['name'], prop_name):
            translate_prop_names.append(prop_name)
        enum = prop.get('enum')
        if enum and len(enum) == 1:
            init_body.append('    self.%(prop_name)s = %(enum)r' % dict(prop_name=prop_name, enum=next(iter(enum))))
        else:
            if prop_name in required:
                if prop_name == 'seq':
                    args.append(prop_name + '=-1')
                else:
                    args.append(prop_name)
            else:
                args.append(prop_name + '=None')
            if prop['type'].__class__ == Ref:
                ref = prop['type']
                ref_data = ref.ref_data
                if ref_data.get('is_enum', False):
                    init_body.append('    if %s is not None:' % (prop_name,))
                    init_body.append('        assert %s in %s.VALID_VALUES' % (prop_name, str(ref)))
                    init_body.append('    self.%(prop_name)s = %(prop_name)s' % dict(prop_name=prop_name))
                else:
                    namespace = dict(prop_name=prop_name, ref_name=str(ref))
                    init_body.append('    if %(prop_name)s is None:' % namespace)
                    init_body.append('        self.%(prop_name)s = %(ref_name)s()' % namespace)
                    init_body.append('    else:')
                    init_body.append('        self.%(prop_name)s = %(ref_name)s(update_ids_from_dap=update_ids_from_dap, **%(prop_name)s) if %(prop_name)s.__class__ !=  %(ref_name)s else %(prop_name)s' % namespace)
            else:
                init_body.append('    self.%(prop_name)s = %(prop_name)s' % dict(prop_name=prop_name))
                if prop['type'] == 'array':
                    ref = prop['items'].get('$ref')
                    if ref is not None:
                        ref_array_cls_name = ref.split('/')[-1]
                        init_body.append('    if update_ids_from_dap and self.%(prop_name)s:' % dict(prop_name=prop_name))
                        init_body.append('        for o in self.%(prop_name)s:' % dict(prop_name=prop_name))
                        init_body.append('            %(ref_array_cls_name)s.update_dict_ids_from_dap(o)' % dict(ref_array_cls_name=ref_array_cls_name))
        prop_type = prop['type']
        prop_description = prop.get('description', '')
        if isinstance(prop_description, (list, tuple)):
            prop_description = '\n    '.join(prop_description)
        docstring.append(':param %(prop_type)s %(prop_name)s: %(prop_description)s' % dict(prop_type=prop_type, prop_name=prop_name, prop_description=prop_description))
    if translate_prop_names:
        init_body.append('    if update_ids_from_dap:')
        for prop_name in translate_prop_names:
            init_body.append('        self.%(prop_name)s = self._translate_id_from_dap(self.%(prop_name)s)' % dict(prop_name=prop_name))
    docstring = _indent_lines('\n'.join(docstring))
    init_body = '\n'.join(init_body)
    args = ', '.join(args)
    if args:
        args = ', ' + args
    class_to_generate['init'] = 'def __init__(self%(args)s, update_ids_from_dap=False, **kwargs):  # noqa (update_ids_from_dap may be unused)\n    """\n%(docstring)s\n    """\n%(init_body)s\n    self.kwargs = kwargs\n' % dict(args=args, init_body=init_body, docstring=docstring)
    class_to_generate['init'] = _indent_lines(class_to_generate['init'])