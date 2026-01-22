from glance.common import exception
from glance.i18n import _
def mutate_image_dict_to_v1(image):
    """
    Replaces a v2-style image dictionary's 'visibility' member with the
    equivalent v1-style 'is_public' member.
    """
    visibility = image.pop('visibility')
    is_image_public = 'public' == visibility
    image['is_public'] = is_image_public
    return image