import codecs
from boto.glacier.exceptions import UploadArchiveError
from boto.glacier.job import Job
from boto.glacier.writer import compute_hashes_from_fileobj, \
from boto.glacier.concurrent import ConcurrentUploader
from boto.glacier.utils import minimum_part_size, DEFAULT_PART_SIZE
import os.path
def list_all_parts(self, upload_id):
    """Automatically make and combine multiple calls to list_parts.

        Call list_parts as necessary, combining the results in case multiple
        calls were required to get data on all available parts.

        """
    result = self.layer1.list_parts(self.name, upload_id)
    marker = result['Marker']
    while marker:
        additional_result = self.layer1.list_parts(self.name, upload_id, marker=marker)
        result['Parts'].extend(additional_result['Parts'])
        marker = additional_result['Marker']
    result['Marker'] = None
    return result