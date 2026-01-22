from mistralclient.api.v2 import client as client_v2
def determine_client_version(mistral_version):
    if mistral_version.find('v2') != -1:
        return 2
    raise RuntimeError('Cannot determine mistral API version')