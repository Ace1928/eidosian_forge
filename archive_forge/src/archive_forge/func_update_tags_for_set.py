import argparse
from osc_lib.i18n import _
def update_tags_for_set(client, obj, parsed_args):
    """Set the tags on an object.

    :param client: The service client to use setting the tags.
    :param obj: The object (Resource) to set the tags on.
    :param parsed_args: Parsed argument object returned by argparse parse_args.
    """
    if parsed_args.no_tag:
        tags = set()
    else:
        tags = set(obj.tags or [])
    if parsed_args.tags:
        tags |= set(parsed_args.tags)
    if set(obj.tags or []) != tags:
        client.set_tags(obj, sorted(list(tags)))