import argcomplete, argparse, requests, pprint
def github_org_members(prefix, parsed_args, **kwargs):
    resource = 'https://api.github.com/orgs/{org}/members'.format(org=parsed_args.organization)
    return (member['login'] for member in requests.get(resource).json() if member['login'].startswith(prefix))