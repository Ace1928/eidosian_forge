from __future__ import (absolute_import, division, print_function)
import re
def parse_docker_image_ref(image_ref, module=None):
    """
        Docker Grammar Reference
        Reference => name [ ":" tag ] [ "@" digest ]
        name => [hostname '/'] component ['/' component]*
            hostname => hostcomponent ['.' hostcomponent]* [':' port-number]
                hostcomponent => /([a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9])/
                port-number   => /[0-9]+/
            component        => alpha-numeric [separator alpha-numeric]*
                alpha-numeric => /[a-z0-9]+/
                separator     => /[_.]|__|[-]*/
    """
    idx = image_ref.find('/')

    def _contains_any(src, values):
        return any((x in src for x in values))
    result = {'tag': None, 'digest': None}
    default_domain = 'docker.io'
    if idx < 0 or (not _contains_any(image_ref[:idx], ':.') and image_ref[:idx] != 'localhost'):
        result['hostname'], remainder = (default_domain, image_ref)
    else:
        result['hostname'], remainder = (image_ref[:idx], image_ref[idx + 1:])
    idx = remainder.find('@')
    if idx > 0 and len(remainder) > idx + 1:
        component, result['digest'] = (remainder[:idx], remainder[idx + 1:])
        err = is_valid_digest(result['digest'])
        if err:
            if module:
                module.fail_json(msg=err)
            return (None, err)
    else:
        idx = remainder.find(':')
        if idx > 0 and len(remainder) > idx + 1:
            component, result['tag'] = (remainder[:idx], remainder[idx + 1:])
        else:
            component = remainder
    v = component.split('/')
    namespace = None
    if len(v) > 1:
        namespace = v[0]
    result.update({'namespace': namespace, 'name': v[-1]})
    return (result, None)