import asyncio
import functools
import queue
import threading
import time
from typing import (
def prepare_response(response: 'CreateArtifactFilesResponseFile') -> ResponsePrepare:
    multipart_resp = response.get('uploadMultipartUrls')
    part_list = multipart_resp['uploadUrlParts'] if multipart_resp else []
    multipart_parts = {u['partNumber']: u['uploadUrl'] for u in part_list} or None
    return ResponsePrepare(birth_artifact_id=response['artifact']['id'], upload_url=response['uploadUrl'], upload_headers=response['uploadHeaders'], upload_id=multipart_resp and multipart_resp.get('uploadID'), storage_path=response.get('storagePath'), multipart_upload_urls=multipart_parts)