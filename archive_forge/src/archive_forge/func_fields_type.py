from openstack import resource
def fields_type(value, resource_type):
    if value is None:
        return None
    resource_mapping = {key: value.name for key, value in resource_type.__dict__.items() if isinstance(value, resource.Body)}
    return comma_separated_list((resource_mapping.get(x, x) for x in value))