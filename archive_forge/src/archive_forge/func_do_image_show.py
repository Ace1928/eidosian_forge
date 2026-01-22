from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
@utils.arg('id', metavar='<uuid>', help='UUID of image to describe.')
def do_image_show(cs, args):
    """Describe a specific image."""
    image = cs.images.get(args.id)
    _show_image(image)