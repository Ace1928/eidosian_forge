from keystoneauth1 import _utils as utils
@property
def media_types(self):
    return self.setdefault('media-types', [])