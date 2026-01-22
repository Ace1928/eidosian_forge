from zunclient.common import cliutils as utils
from zunclient.common import utils as zun_utils
@utils.arg('--marker', metavar='<marker>', default=None, help='The last image UUID of the previous page; displays list of images after "marker".')
@utils.arg('--limit', metavar='<limit>', type=int, help='Maximum number of images to return')
@utils.arg('--sort-key', metavar='<sort-key>', help='Column to sort results by')
@utils.arg('--sort-dir', metavar='<sort-dir>', choices=['desc', 'asc'], help='Direction to sort. "asc" or "desc".')
def do_image_list(cs, args):
    """Print a list of available images."""
    opts = {}
    opts['marker'] = args.marker
    opts['limit'] = args.limit
    opts['sort_key'] = args.sort_key
    opts['sort_dir'] = args.sort_dir
    opts = zun_utils.remove_null_parms(**opts)
    images = cs.images.list(**opts)
    columns = ('uuid', 'image_id', 'repo', 'tag', 'size')
    utils.print_list(images, columns, {'versions': zun_utils.print_list_field('versions')}, sortby_index=None)