from cinderclient import api_versions
from cinderclient import utils
@api_versions.wraps('3.2', '3.3')
def do_fake_action():
    return 'fake_action 3.2 to 3.3'