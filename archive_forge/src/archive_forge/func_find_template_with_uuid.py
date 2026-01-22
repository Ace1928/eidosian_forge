import time
def find_template_with_uuid(uuid, templates):
    return next((template for template in templates if template['uuid'] == uuid), None)