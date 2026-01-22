from saharaclient.api import base
def unregister_image(self, image_id):
    """Remove an Image from Sahara Image Registry."""
    self._delete('/images/%s' % image_id)