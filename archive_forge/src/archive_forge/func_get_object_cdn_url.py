from libcloud.storage.drivers.s3 import (
def get_object_cdn_url(self, obj, ex_expiry=S3_CDN_URL_EXPIRY_HOURS):
    return S3StorageDriver.get_object_cdn_url(self, obj, ex_expiry=ex_expiry)