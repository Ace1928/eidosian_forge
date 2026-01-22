from typing import Any, Dict, List, Optional, Union
from .. import config
from ..exceptions import DatasetsError
from .file_utils import (
from .logging import get_logger
def get_exported_dataset_infos(dataset: str, revision: str, token: Optional[Union[str, bool]]) -> Dict[str, Dict[str, Any]]:
    """
    Get the dataset information, can be useful to get e.g. the dataset features.
    Docs: https://huggingface.co/docs/datasets-server/info
    """
    datasets_server_info_url = config.HF_ENDPOINT.replace('://', '://datasets-server.') + '/info?dataset='
    try:
        info_response = http_get(url=datasets_server_info_url + dataset, temp_file=None, headers=get_authentication_headers_for_url(config.HF_ENDPOINT + f'datasets/{dataset}', token=token), timeout=100.0, max_retries=3)
        info_response.raise_for_status()
        if 'X-Revision' in info_response.headers:
            if info_response.headers['X-Revision'] == revision or revision is None:
                info_response = info_response.json()
                if info_response.get('partial') is False and (not info_response.get('pending', True)) and (not info_response.get('failed', True)) and ('dataset_info' in info_response):
                    return info_response['dataset_info']
                else:
                    logger.debug(f'Dataset info for {dataset} is not completely ready yet.')
            else:
                logger.debug(f"Dataset info for {dataset} is available but outdated (revision='{info_response.headers['X-Revision']}')")
    except Exception as e:
        logger.debug(f'No dataset info for {dataset} available ({type(e).__name__}: {e})')
    raise DatasetsServerError('No exported dataset infos available.')