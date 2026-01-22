import json
import logging
import numbers
from hashlib import sha1
from typing import Any, Dict, Iterable, List, Optional, Tuple
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
def delete_documents_with_document_id(self, id_list: List[str]) -> bool:
    """Delete documents based on their IDs.

        Args:
            id_list: List of document IDs.
        Returns:
            Whether the deletion was successful or not.
        """
    if id_list is None or len(id_list) == 0:
        return True
    from alibabacloud_ha3engine_vector import models
    delete_doc_list = []
    for doc_id in id_list:
        delete_doc_list.append({'fields': {self.config.field_name_mapping['id']: doc_id}, 'cmd': 'delete'})
    delete_request = models.PushDocumentsRequest(self.options_headers, delete_doc_list)
    try:
        delete_response = self.ha3_engine_client.push_documents(self.config.opt_table_name, self.config.field_name_mapping['id'], delete_request)
        json_response = json.loads(delete_response.body)
        return json_response['status'] == 'OK'
    except Exception as e:
        logger.error(f'delete doc from :{self.config.endpoint} instance_id:{self.config.instance_id} failed.', e)
        raise e