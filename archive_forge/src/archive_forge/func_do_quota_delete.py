from zunclient.common import cliutils as utils
@utils.arg('project_id', metavar='<project_id>', help='The UUID of project in a multi-project cloud')
def do_quota_delete(cs, args):
    """Delete quotas for a project"""
    cs.quotas.delete(args.project_id)