from boto.cloudfront.identity import OriginAccessIdentity
def get_oai_value(origin_access_identity):
    if isinstance(origin_access_identity, OriginAccessIdentity):
        return origin_access_identity.uri()
    else:
        return origin_access_identity