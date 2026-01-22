from os.path import exists
import sys
from collections import defaultdict
import json
def process_changelog(filename_in, filename_out):
    if exists(filename_out):
        raise ValueError('{} already exists and would be overwritten'.format(filename_out))
    with open(filename_in, 'r') as fh:
        data = json.load(fh)
    prs = data['data']['repository']['milestone']['pullRequests']['nodes']
    bad_pr = False
    grouped = defaultdict(list)
    highlighted = []
    api_breaks = []
    deprecates = []
    for item in prs:
        n = item['number']
        title = item['title']
        labels = [label['name'] for label in item['labels']['nodes']]
        api_break = 'Notes: API-break' in labels
        highlight = 'Notes: Release-highlight' in labels
        deprecated = 'Notes: API-deprecation' in labels
        component_str = 'Component: '
        components = [label[len(component_str):] for label in labels if label.startswith(component_str)]
        if not components:
            print(f'Found no component label for #{n}')
            bad_pr = True
            continue
        if len(components) > 1:
            print(f'Found more than one component label for #{n}')
            bad_pr = True
            continue
        grouped[components[0]].append((n, title))
        if highlight:
            highlighted.append((n, title))
        if api_break:
            api_breaks.append((n, title))
        if deprecated:
            deprecates.append((n, title))
    if bad_pr:
        raise ValueError('One or more PRs have no, or more than one component label')
    with open(filename_out, 'w') as fh:
        write_special_section(fh, highlighted, 'Highlights')
        write_special_section(fh, deprecates, 'Deprecated')
        write_special_section(fh, api_breaks, 'Breaking changes')
        for group, items in sorted(grouped.items(), key=lambda x: x[0]):
            write_special_section(fh, items, group.capitalize())