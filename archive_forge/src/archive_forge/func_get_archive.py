from datetime import datetime
from .. import errors
from .. import utils
from ..constants import DEFAULT_DATA_CHUNK_SIZE
from ..types import CancellableStream
from ..types import ContainerConfig
from ..types import EndpointConfig
from ..types import HostConfig
from ..types import NetworkingConfig
@utils.check_resource('container')
def get_archive(self, container, path, chunk_size=DEFAULT_DATA_CHUNK_SIZE, encode_stream=False):
    """
        Retrieve a file or folder from a container in the form of a tar
        archive.

        Args:
            container (str): The container where the file is located
            path (str): Path to the file or folder to retrieve
            chunk_size (int): The number of bytes returned by each iteration
                of the generator. If ``None``, data will be streamed as it is
                received. Default: 2 MB
            encode_stream (bool): Determines if data should be encoded
                (gzip-compressed) during transmission. Default: False

        Returns:
            (tuple): First element is a raw tar data stream. Second element is
            a dict containing ``stat`` information on the specified ``path``.

        Raises:
            :py:class:`docker.errors.APIError`
                If the server returns an error.

        Example:

            >>> c = docker.APIClient()
            >>> f = open('./sh_bin.tar', 'wb')
            >>> bits, stat = c.api.get_archive(container, '/bin/sh')
            >>> print(stat)
            {'name': 'sh', 'size': 1075464, 'mode': 493,
             'mtime': '2018-10-01T15:37:48-07:00', 'linkTarget': ''}
            >>> for chunk in bits:
            ...    f.write(chunk)
            >>> f.close()
        """
    params = {'path': path}
    headers = {'Accept-Encoding': 'gzip, deflate'} if encode_stream else {'Accept-Encoding': 'identity'}
    url = self._url('/containers/{0}/archive', container)
    res = self._get(url, params=params, stream=True, headers=headers)
    self._raise_for_status(res)
    encoded_stat = res.headers.get('x-docker-container-path-stat')
    return (self._stream_raw_result(res, chunk_size, False), utils.decode_json_header(encoded_stat) if encoded_stat else None)