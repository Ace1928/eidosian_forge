from ncclient.xml_ import *
from ncclient.operations.errors import OperationError, MissingCapabilityError
def url_validator(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False