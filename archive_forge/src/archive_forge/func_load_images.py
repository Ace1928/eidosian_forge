from __future__ import absolute_import, division, print_function
import errno
import traceback
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import DockerException
def load_images(self):
    """
        Load images from a .tar archive
        """
    load_output = []
    try:
        self.log('Opening image {0}'.format(self.path))
        with open(self.path, 'rb') as image_tar:
            self.log('Loading images from {0}'.format(self.path))
            res = self.client._post(self.client._url('/images/load'), data=image_tar, stream=True)
            for line in self.client._stream_helper(res, decode=True):
                self.log(line, pretty_print=True)
                self._extract_output_line(line, load_output)
    except EnvironmentError as exc:
        if exc.errno == errno.ENOENT:
            self.client.fail('Error opening archive {0} - {1}'.format(self.path, to_native(exc)))
        self.client.fail('Error loading archive {0} - {1}'.format(self.path, to_native(exc)), stdout='\n'.join(load_output))
    except Exception as exc:
        self.client.fail('Error loading archive {0} - {1}'.format(self.path, to_native(exc)), stdout='\n'.join(load_output))
    loaded_images = []
    for line in load_output:
        if line.startswith('Loaded image:'):
            loaded_images.append(line[len('Loaded image:'):].strip())
        if line.startswith('Loaded image ID:'):
            loaded_images.append(line[len('Loaded image ID:'):].strip())
    if not loaded_images:
        self.client.fail('Detected no loaded images. Archive potentially corrupt?', stdout='\n'.join(load_output))
    images = []
    for image_name in loaded_images:
        if is_image_name_id(image_name):
            images.append(self.client.find_image_by_id(image_name))
        elif ':' in image_name:
            image_name, tag = image_name.rsplit(':', 1)
            images.append(self.client.find_image(image_name, tag))
        else:
            self.client.module.warn('Image name "{0}" is neither ID nor has a tag'.format(image_name))
    self.results['image_names'] = loaded_images
    self.results['images'] = images
    self.results['changed'] = True
    self.results['stdout'] = '\n'.join(load_output)