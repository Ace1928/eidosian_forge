import pytest
from datetime import timedelta
import pyarrow as pa
def validate_kms_connection_config(kms_connection_config):
    assert 'Instance1' == kms_connection_config.kms_instance_id
    assert 'URL1' == kms_connection_config.kms_instance_url
    assert 'MyToken' == kms_connection_config.key_access_token
    assert {'key1': 'key_material_1', 'key2': 'key_material_2'} == kms_connection_config.custom_kms_conf