import glob
import logging
import os
import re
from typing import List, Tuple
import requests
def fetch_external_assets(docs_folder: str='docs/source', assets_folder: str='fetched-s3-assets', file_pattern: str='*.rst', retrieve_pattern: str='https?://[-a-zA-Z0-9_]+\\.s3\\.[-a-zA-Z0-9()_\\\\+.\\\\/=]+') -> None:
    """Search all URL in docs, download these files locally and replace online with local version.

    Args:
        docs_folder: the location of docs related to the project root
        assets_folder: a folder inside ``docs_folder`` to be created and saving online assets
        file_pattern: what kind of files shall be scanned
        retrieve_pattern: pattern for reg. expression to search URL/S3 resources

    """
    list_files = glob.glob(os.path.join(docs_folder, '**', file_pattern), recursive=True)
    if not list_files:
        logging.warning(f'no files were listed in folder "{docs_folder}" and pattern "{file_pattern}"')
        return
    urls = _search_all_occurrences(list_files, pattern=retrieve_pattern)
    if not urls:
        logging.info(f'no resources/assets were match in {docs_folder} for {retrieve_pattern}')
        return
    target_folder = os.path.join(docs_folder, assets_folder)
    os.makedirs(target_folder, exist_ok=True)
    pairs_url_file = []
    for i, url in enumerate(set(urls)):
        logging.info(f' >> downloading ({i}/{len(urls)}): {url}')
        fname = _download_file(url, target_folder)
        pairs_url_file.append((url, os.path.join(assets_folder, fname)))
    for fpath in list_files:
        _replace_remote_with_local(fpath, docs_folder, pairs_url_file)