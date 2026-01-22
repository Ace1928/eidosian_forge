from __future__ import absolute_import, division, print_function
def roledefinition_to_dict(role):
    result = dict(id=role.id, name=role.name, type=role.role_type, assignable_scopes=role.assignable_scopes, description=role.description, role_name=role.role_name)
    if role.permissions:
        result['permissions'] = [dict(actions=p.actions, not_actions=p.not_actions, data_actions=p.data_actions, not_data_actions=p.not_data_actions) for p in role.permissions]
    return result