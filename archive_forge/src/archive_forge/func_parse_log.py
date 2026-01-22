from datetime import datetime, timedelta
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def parse_log(self, log: dict) -> Document:
    """
        Create Document objects from Datadog log items.
        """
    attributes = log.get('attributes', {})
    metadata = {'id': log.get('id', ''), 'status': attributes.get('status'), 'service': attributes.get('service', ''), 'tags': attributes.get('tags', []), 'timestamp': attributes.get('timestamp', '')}
    message = attributes.get('message', '')
    inside_attributes = attributes.get('attributes', {})
    content_dict = {**inside_attributes, 'message': message}
    content = ', '.join((f'{k}: {v}' for k, v in content_dict.items()))
    return Document(page_content=content, metadata=metadata)