import logging
import uuid
def calculate_uuid_ref(ref, entity):
    entity_uuid = validate_ref_and_return_uuid(ref, entity.capitalize().rstrip('s'))
    entity_ref = '{entity}/{uuid}'.format(entity=entity, uuid=entity_uuid)
    LOG.info('Calculated %s uuid ref: %s', entity.capitalize(), entity_ref)
    return entity_ref