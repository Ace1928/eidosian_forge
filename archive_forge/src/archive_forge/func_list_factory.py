from cinderclient import api_versions
from cinderclient.apiclient import base as common_base
from cinderclient import base
@classmethod
def list_factory(cls, mngr, elements):
    return [cls(mngr, element, loaded=True) for element in elements]