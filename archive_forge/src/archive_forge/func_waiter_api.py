from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.client import (
from botocore.docs.paginator import PaginatorDocumenter
from botocore.docs.waiter import WaiterDocumenter
from botocore.exceptions import DataNotFoundError
def waiter_api(self, section):
    if self._client.waiter_names:
        service_waiter_model = self._session.get_waiter_model(self._service_name)
        waiter_documenter = WaiterDocumenter(self._client, service_waiter_model, self._root_docs_path)
        waiter_documenter.document_waiters(section)