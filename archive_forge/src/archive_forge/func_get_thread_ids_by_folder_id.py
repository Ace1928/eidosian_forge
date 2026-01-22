import logging
import re
import xml.etree.cElementTree
import xml.sax.saxutils
from io import BytesIO
from typing import List, Optional, Sequence
from xml.etree.ElementTree import ElementTree
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
def get_thread_ids_by_folder_id(self, folder_id: str, depth: int, thread_ids: List[str]) -> None:
    """Get thread ids by folder id and update in thread_ids"""
    from quip_api.quip import HTTPError, QuipError
    try:
        folder = self.quip_client.get_folder(folder_id)
    except QuipError as e:
        if e.code == 403:
            logging.warning(f'depth {depth}, Skipped over restricted folder {folder_id}, {e}')
        else:
            logging.warning(f'depth {depth}, Skipped over folder {folder_id} due to unknown error {e.code}')
        return
    except HTTPError as e:
        logging.warning(f'depth {depth}, Skipped over folder {folder_id} due to HTTP error {e.code}')
        return
    title = folder['folder'].get('title', 'Folder %s' % folder_id)
    logging.info(f'depth {depth}, Processing folder {title}')
    for child in folder['children']:
        if 'folder_id' in child:
            self.get_thread_ids_by_folder_id(child['folder_id'], depth + 1, thread_ids)
        elif 'thread_id' in child:
            thread_ids.append(child['thread_id'])