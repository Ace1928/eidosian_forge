from ..objecttype import ObjectType
from ..schema import Schema
from ..uuid import UUID
from ..structures import NonNull
def resolve_required_uuid(self, info, input):
    return input