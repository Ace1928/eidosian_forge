def yaqlize(class_or_object=None, yaqlize_attributes=True, yaqlize_methods=True, yaqlize_indexer=True, auto_yaqlize_result=False, whitelist=None, blacklist=None, attribute_remapping=None, blacklist_remapped_attributes=True):

    def func(something):
        if not hasattr(something, YAQLIZATION_ATTR):
            setattr(something, YAQLIZATION_ATTR, build_yaqlization_settings(yaqlize_attributes=yaqlize_attributes, yaqlize_methods=yaqlize_methods, yaqlize_indexer=yaqlize_indexer, auto_yaqlize_result=auto_yaqlize_result, whitelist=whitelist, blacklist=blacklist, attribute_remapping=attribute_remapping))
        return something
    if class_or_object is None:
        return func
    else:
        return func(class_or_object)