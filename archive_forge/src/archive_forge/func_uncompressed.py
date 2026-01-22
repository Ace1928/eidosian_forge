from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import errno
import io
import json
import os
import tarfile
import concurrent.futures
from containerregistry.client import docker_name
from containerregistry.client.v1 import docker_image as v1_image
from containerregistry.client.v1 import save as v1_save
from containerregistry.client.v2 import v1_compat
from containerregistry.client.v2_2 import docker_digest
from containerregistry.client.v2_2 import docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import v2_compat
import six
def uncompressed(image, directory, threads=1):
    """Produce a format similar to `fast()`, but with uncompressed blobs.

  After calling this, the following filesystem will exist:
    directory/
      config.json   <-- only *.json, the image's config
      digest        <-- sha256 digest of the image's manifest
      manifest.json <-- the image's manifest
      001.tar       <-- the first layer's .tar filesystem delta
      001.sha256    <-- the sha256 of 001.tar with a "sha256:" prefix.
      ...
      NNN.tar       <-- the NNNth layer's .tar filesystem delta
      NNN.sha256    <-- the sha256 of NNN.tar with a "sha256:" prefix.

  We pad layer indices to only 3 digits because of a known ceiling on the number
  of filesystem layers Docker supports.

  Args:
    image: a docker image to save.
    directory: an existing empty directory under which to save the layout.
    threads: the number of threads to use when performing the upload.

  Returns:
    A tuple whose first element is the path to the config file, and whose second
    element is an ordered list of tuples whose elements are the filenames
    containing: (.sha256, .tar) respectively.
  """

    def write_file(name, accessor, arg):
        with io.open(name, u'wb') as f:
            f.write(accessor(arg))
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
        future_to_params = {}
        config_file = os.path.join(directory, 'config.json')
        f = executor.submit(write_file, config_file, lambda unused: image.config_file().encode('utf8'), 'unused')
        future_to_params[f] = config_file
        executor.submit(write_file, os.path.join(directory, 'digest'), lambda unused: image.digest().encode('utf8'), 'unused')
        executor.submit(write_file, os.path.join(directory, 'manifest.json'), lambda unused: image.manifest().encode('utf8'), 'unused')
        idx = 0
        layers = []
        for diff_id in reversed(image.diff_ids()):
            digest_name = os.path.join(directory, '%03d.sha256' % idx)
            f = executor.submit(write_file, digest_name, lambda diff_id: diff_id[7:].encode('utf8'), diff_id)
            future_to_params[f] = digest_name
            layer_name = os.path.join(directory, '%03d.tar' % idx)
            f = executor.submit(write_file, layer_name, image.uncompressed_layer, diff_id)
            future_to_params[f] = layer_name
            layers.append((digest_name, layer_name))
            idx += 1
        for future in concurrent.futures.as_completed(future_to_params):
            future.result()
    return (config_file, layers)