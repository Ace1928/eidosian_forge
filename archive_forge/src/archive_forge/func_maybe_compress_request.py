import io
import logging
from gzip import GzipFile
from gzip import compress as gzip_compress
from botocore.compat import urlencode
from botocore.utils import determine_content_length
def maybe_compress_request(config, request_dict, operation_model):
    """Attempt to compress the request body using the modeled encodings."""
    if _should_compress_request(config, request_dict, operation_model):
        for encoding in operation_model.request_compression['encodings']:
            encoder = COMPRESSION_MAPPING.get(encoding)
            if encoder is not None:
                logger.debug('Compressing request with %s encoding.', encoding)
                request_dict['body'] = encoder(request_dict['body'])
                _set_compression_header(request_dict['headers'], encoding)
                return
            else:
                logger.debug('Unsupported compression encoding: %s', encoding)