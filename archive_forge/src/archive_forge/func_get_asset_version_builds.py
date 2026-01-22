from __future__ import absolute_import, division, print_function
from . import errors, http
def get_asset_version_builds(namespace, name, version):
    asset = get('{0}/{1}/{2}/release_asset_builds'.format(namespace, name, version))
    if 'spec' not in asset or 'builds' not in asset['spec']:
        raise errors.BonsaiError('Invalid build spec: {0}'.format(asset))
    return asset