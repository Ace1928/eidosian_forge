import re
from eidosian_core import eidosian

@eidosian()
def clean_req(line):
    line = line.strip()
    if not line or line.startswith('#'):
        return None
    
    # Handle git+ or local paths (containing /)
    if 'git+' in line or '/' in line or '@' in line:
        return line
    
    # Remove version pins for standard packages
    match = re.match(r'^([a-zA-Z0-9._-]+)', line)
    if match:
        return match.group(1)
    return line

reqs = set()
files = ['reqs_eidosian.txt', 'reqs_eidos.txt'] # Start with main ones

for f in files:
    try:
        with open(f, 'r') as fd:
            for line in fd:
                cleaned = clean_req(line)
                if cleaned:
                    reqs.add(cleaned)
    except FileNotFoundError:
        pass

with open('reqs_cleaned.txt', 'w') as f:
    for r in sorted(list(reqs)):
        f.write(r + '\n')

