import os
import pkg_resources
from urllib.parse import quote
import string
import inspect
def query_interactive(src_fn, dest_fn, src_content, dest_content, simulate):
    global all_answer
    from difflib import unified_diff, context_diff
    u_diff = list(unified_diff(dest_content.splitlines(), src_content.splitlines(), dest_fn, src_fn))
    c_diff = list(context_diff(dest_content.splitlines(), src_content.splitlines(), dest_fn, src_fn))
    added = len([l for l in u_diff if l.startswith('+') and (not l.startswith('+++'))])
    removed = len([l for l in u_diff if l.startswith('-') and (not l.startswith('---'))])
    if added > removed:
        msg = '; %i lines added' % (added - removed)
    elif removed > added:
        msg = '; %i lines removed' % (removed - added)
    else:
        msg = ''
    print('Replace %i bytes with %i bytes (%i/%i lines changed%s)' % (len(dest_content), len(src_content), removed, len(dest_content.splitlines()), msg))
    prompt = 'Overwrite %s [y/n/d/B/?] ' % dest_fn
    while 1:
        if all_answer is None:
            response = input(prompt).strip().lower()
        else:
            response = all_answer
        if not response or response[0] == 'b':
            import shutil
            new_dest_fn = dest_fn + '.bak'
            n = 0
            while os.path.exists(new_dest_fn):
                n += 1
                new_dest_fn = dest_fn + '.bak' + str(n)
            print('Backing up %s to %s' % (dest_fn, new_dest_fn))
            if not simulate:
                shutil.copyfile(dest_fn, new_dest_fn)
            return True
        elif response.startswith('all '):
            rest = response[4:].strip()
            if not rest or rest[0] not in ('y', 'n', 'b'):
                print(query_usage)
                continue
            response = all_answer = rest[0]
        if response[0] == 'y':
            return True
        elif response[0] == 'n':
            return False
        elif response == 'dc':
            print('\n'.join(c_diff))
        elif response[0] == 'd':
            print('\n'.join(u_diff))
        else:
            print(query_usage)