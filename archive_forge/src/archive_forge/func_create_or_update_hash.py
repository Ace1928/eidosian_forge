from ._base import *
from .types import CreateSchemaType, UpdateSchemaType
from .models import LazyHasher, LazyUserSchema, LazyDBConfig, LazyDBSaveMetrics
def create_or_update_hash(self, data=None, update_data=None, item=None):
    if not self.has_hashing:
        if data:
            return data
        return (item, update_data)
    logger.info(self.hash_schema)
    for field, hash_field in self.hash_schema.items():
        if data and data.get(field):
            data[hash_field] = self.create_hash(val=data[field])
            if self.is_prod:
                _ = data.pop(field)
        elif update_data and update_data.get(field):
            item, updated = self.update_hash(prop=field, hash_prop=hash_field, new_val=update_data[field], do_verify=True, item=item)
            if updated:
                logger.info(f'[{self.index_name} Index]: Updated Hashed Field: {field} = {hash_field}')
                if not self.is_prod:
                    setattr(item, field, update_data[field])
            else:
                logger.error(f'[{self.index_name} Index]: Failed Validation for Hash Field {field} = {hash_field}')
            _ = update_data.pop(field)
    if data:
        return data
    return (item, update_data)