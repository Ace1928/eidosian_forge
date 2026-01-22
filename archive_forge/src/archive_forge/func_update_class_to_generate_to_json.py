def update_class_to_generate_to_json(class_to_generate):
    required = _OrderedSet(class_to_generate.get('required', _OrderedSet()))
    prop_name_and_prop = extract_prop_name_and_prop(class_to_generate)
    to_dict_body = ['def to_dict(self, update_ids_to_dap=False):  # noqa (update_ids_to_dap may be unused)']
    translate_prop_names = []
    for prop_name, prop in prop_name_and_prop:
        if is_variable_to_translate(class_to_generate['name'], prop_name):
            translate_prop_names.append(prop_name)
    for prop_name, prop in prop_name_and_prop:
        namespace = dict(prop_name=prop_name, noqa=_get_noqa_for_var(prop_name))
        to_dict_body.append('    %(prop_name)s = self.%(prop_name)s%(noqa)s' % namespace)
        if prop.get('type') == 'array':
            to_dict_body.append('    if %(prop_name)s and hasattr(%(prop_name)s[0], "to_dict"):' % namespace)
            to_dict_body.append('        %(prop_name)s = [x.to_dict() for x in %(prop_name)s]' % namespace)
    if translate_prop_names:
        to_dict_body.append('    if update_ids_to_dap:')
        for prop_name in translate_prop_names:
            namespace = dict(prop_name=prop_name, noqa=_get_noqa_for_var(prop_name))
            to_dict_body.append('        if %(prop_name)s is not None:' % namespace)
            to_dict_body.append('            %(prop_name)s = self._translate_id_to_dap(%(prop_name)s)%(noqa)s' % namespace)
    if not translate_prop_names:
        update_dict_ids_from_dap_body = []
    else:
        update_dict_ids_from_dap_body = ['', '', '@classmethod', 'def update_dict_ids_from_dap(cls, dct):']
        for prop_name in translate_prop_names:
            namespace = dict(prop_name=prop_name)
            update_dict_ids_from_dap_body.append('    if %(prop_name)r in dct:' % namespace)
            update_dict_ids_from_dap_body.append('        dct[%(prop_name)r] = cls._translate_id_from_dap(dct[%(prop_name)r])' % namespace)
        update_dict_ids_from_dap_body.append('    return dct')
    class_to_generate['update_dict_ids_from_dap'] = _indent_lines('\n'.join(update_dict_ids_from_dap_body))
    to_dict_body.append('    dct = {')
    first_not_required = False
    for prop_name, prop in prop_name_and_prop:
        use_to_dict = prop['type'].__class__ == Ref and (not prop['type'].ref_data.get('is_enum', False))
        is_array = prop['type'] == 'array'
        ref_array_cls_name = ''
        if is_array:
            ref = prop['items'].get('$ref')
            if ref is not None:
                ref_array_cls_name = ref.split('/')[-1]
        namespace = dict(prop_name=prop_name, ref_array_cls_name=ref_array_cls_name)
        if prop_name in required:
            if use_to_dict:
                to_dict_body.append('        %(prop_name)r: %(prop_name)s.to_dict(update_ids_to_dap=update_ids_to_dap),' % namespace)
            elif ref_array_cls_name:
                to_dict_body.append('        %(prop_name)r: [%(ref_array_cls_name)s.update_dict_ids_to_dap(o) for o in %(prop_name)s] if (update_ids_to_dap and %(prop_name)s) else %(prop_name)s,' % namespace)
            else:
                to_dict_body.append('        %(prop_name)r: %(prop_name)s,' % namespace)
        else:
            if not first_not_required:
                first_not_required = True
                to_dict_body.append('    }')
            to_dict_body.append('    if %(prop_name)s is not None:' % namespace)
            if use_to_dict:
                to_dict_body.append('        dct[%(prop_name)r] = %(prop_name)s.to_dict(update_ids_to_dap=update_ids_to_dap)' % namespace)
            elif ref_array_cls_name:
                to_dict_body.append('        dct[%(prop_name)r] = [%(ref_array_cls_name)s.update_dict_ids_to_dap(o) for o in %(prop_name)s] if (update_ids_to_dap and %(prop_name)s) else %(prop_name)s' % namespace)
            else:
                to_dict_body.append('        dct[%(prop_name)r] = %(prop_name)s' % namespace)
    if not first_not_required:
        first_not_required = True
        to_dict_body.append('    }')
    to_dict_body.append('    dct.update(self.kwargs)')
    to_dict_body.append('    return dct')
    class_to_generate['to_dict'] = _indent_lines('\n'.join(to_dict_body))
    if not translate_prop_names:
        update_dict_ids_to_dap_body = []
    else:
        update_dict_ids_to_dap_body = ['', '', '@classmethod', 'def update_dict_ids_to_dap(cls, dct):']
        for prop_name in translate_prop_names:
            namespace = dict(prop_name=prop_name)
            update_dict_ids_to_dap_body.append('    if %(prop_name)r in dct:' % namespace)
            update_dict_ids_to_dap_body.append('        dct[%(prop_name)r] = cls._translate_id_to_dap(dct[%(prop_name)r])' % namespace)
        update_dict_ids_to_dap_body.append('    return dct')
    class_to_generate['update_dict_ids_to_dap'] = _indent_lines('\n'.join(update_dict_ids_to_dap_body))