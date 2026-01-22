import pytest
from datetime import timedelta
import pyarrow as pa
def test_encrypted_parquet_kms_configuration():

    def validate_kms_connection_config(kms_connection_config):
        assert 'Instance1' == kms_connection_config.kms_instance_id
        assert 'URL1' == kms_connection_config.kms_instance_url
        assert 'MyToken' == kms_connection_config.key_access_token
        assert {'key1': 'key_material_1', 'key2': 'key_material_2'} == kms_connection_config.custom_kms_conf
    kms_connection_config = pe.KmsConnectionConfig(kms_instance_id='Instance1', kms_instance_url='URL1', key_access_token='MyToken', custom_kms_conf={'key1': 'key_material_1', 'key2': 'key_material_2'})
    validate_kms_connection_config(kms_connection_config)
    kms_connection_config_1 = pe.KmsConnectionConfig()
    kms_connection_config_1.kms_instance_id = 'Instance1'
    kms_connection_config_1.kms_instance_url = 'URL1'
    kms_connection_config_1.key_access_token = 'MyToken'
    kms_connection_config_1.custom_kms_conf = {'key1': 'key_material_1', 'key2': 'key_material_2'}
    validate_kms_connection_config(kms_connection_config_1)