from botocore.docs.bcdoc.restdoc import DocumentStructure
from botocore.docs.client import (
from botocore.docs.paginator import PaginatorDocumenter
from botocore.docs.waiter import WaiterDocumenter
from botocore.exceptions import DataNotFoundError
def client_exceptions(self, section):
    ClientExceptionsDocumenter(self._client, self._root_docs_path).document_exceptions(section)