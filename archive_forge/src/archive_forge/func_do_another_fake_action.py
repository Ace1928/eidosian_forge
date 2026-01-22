from cinderclient import api_versions
from cinderclient import utils
@api_versions.wraps('3.6')
@utils.arg('--foo', start_version='3.7')
def do_another_fake_action():
    return 'another_fake_action'