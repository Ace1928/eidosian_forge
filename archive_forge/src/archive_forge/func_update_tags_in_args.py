import argparse
from openstackclient.i18n import _
def update_tags_in_args(parsed_args, obj, args):
    if parsed_args.clear_tags:
        args['tags'] = []
        obj.tags = []
    if parsed_args.remove_tag:
        args['tags'] = list(set(obj.tags) - set(parsed_args.remove_tag))
        return
    if parsed_args.tags:
        args['tags'] = list(set(obj.tags).union(set(parsed_args.tags)))