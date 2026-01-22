from ._base import *
from .models import Params, component_name
from .exceptions import *
from .dependencies import fix_query_dependencies, clone_dependant, insert_dependencies
def get_jsonrpc_request_id() -> Optional[Union[str, int]]:
    return get_jsonrpc_context().raw_request.get('id')