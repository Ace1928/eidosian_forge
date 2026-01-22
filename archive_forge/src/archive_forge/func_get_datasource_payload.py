from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible.module_utils.urls import fetch_url, basic_auth_header
from ansible_collections.community.grafana.plugins.module_utils import base
def get_datasource_payload(data, org_id=None):
    payload = {'orgId': data['org_id'] if org_id is None else org_id, 'name': data['name'], 'uid': data['uid'], 'type': data['ds_type'], 'access': data['access'], 'url': data['ds_url'], 'database': data['database'], 'withCredentials': data['with_credentials'], 'isDefault': data['is_default'], 'user': data['user'], 'jsonData': data['additional_json_data'], 'secureJsonData': data['additional_secure_json_data']}
    json_data = payload['jsonData']
    secure_json_data = payload['secureJsonData']
    if data.get('password'):
        secure_json_data['password'] = data['password']
    if 'basic_auth_user' in data and data['basic_auth_user'] and ('basic_auth_password' in data) and data['basic_auth_password']:
        payload['basicAuth'] = True
        payload['basicAuthUser'] = data['basic_auth_user']
        secure_json_data['basicAuthPassword'] = data['basic_auth_password']
    else:
        payload['basicAuth'] = False
    if data.get('tls_client_cert') and data.get('tls_client_key'):
        json_data['tlsAuth'] = True
        if data.get('tls_ca_cert'):
            secure_json_data['tlsCACert'] = data['tls_ca_cert']
            secure_json_data['tlsClientCert'] = data['tls_client_cert']
            secure_json_data['tlsClientKey'] = data['tls_client_key']
            json_data['tlsAuthWithCACert'] = True
        else:
            secure_json_data['tlsClientCert'] = data['tls_client_cert']
            secure_json_data['tlsClientKey'] = data['tls_client_key']
    else:
        json_data['tlsAuth'] = False
        json_data['tlsAuthWithCACert'] = False
        if data.get('tls_ca_cert'):
            json_data['tlsAuthWithCACert'] = True
            secure_json_data['tlsCACert'] = data['tls_ca_cert']
    if data.get('tls_skip_verify'):
        json_data['tlsSkipVerify'] = True
    if data['ds_type'] == 'elasticsearch':
        json_data['maxConcurrentShardRequests'] = data['max_concurrent_shard_requests']
        json_data['timeField'] = data['time_field']
        if data.get('interval'):
            json_data['interval'] = data['interval']
        try:
            es_version = int(data['es_version'])
            if es_version < 56:
                json_data.pop('maxConcurrentShardRequests')
        except ValueError:
            es_version = ES_VERSION_MAPPING.get(data['es_version'])
        json_data['esVersion'] = es_version
    if data['ds_type'] in ['elasticsearch', 'influxdb', 'prometheus']:
        if data.get('time_interval'):
            json_data['timeInterval'] = data['time_interval']
    if data['ds_type'] == 'opentsdb':
        json_data['tsdbVersion'] = data['tsdb_version']
        if data['tsdb_resolution'] == 'second':
            json_data['tsdbResolution'] = 1
        else:
            json_data['tsdbResolution'] = 2
    if data['ds_type'] == 'postgres':
        json_data['sslmode'] = data['sslmode']
    if data['ds_type'] == 'alexanderzobnin-zabbix-datasource':
        if data.get('trends'):
            json_data['trends'] = True
        json_data['username'] = data['zabbix_user']
        json_data['password'] = data['zabbix_password']
    if data['ds_type'] == 'grafana-azure-monitor-datasource':
        json_data['tenantId'] = data['azure_tenant']
        json_data['clientId'] = data['azure_client']
        json_data['cloudName'] = data['azure_cloud']
        json_data['clientsecret'] = 'clientsecret'
        if data.get('azure_secret'):
            secure_json_data['clientSecret'] = data['azure_secret']
    if data['ds_type'] == 'cloudwatch':
        if data.get('aws_credentials_profile'):
            payload['database'] = data.get('aws_credentials_profile')
        json_data['authType'] = data['aws_auth_type']
        json_data['defaultRegion'] = data['aws_default_region']
        if data.get('aws_custom_metrics_namespaces'):
            json_data['customMetricsNamespaces'] = data.get('aws_custom_metrics_namespaces')
        if data.get('aws_assume_role_arn'):
            json_data['assumeRoleArn'] = data.get('aws_assume_role_arn')
        if data.get('aws_access_key') and data.get('aws_secret_key'):
            secure_json_data['accessKey'] = data.get('aws_access_key')
            secure_json_data['secretKey'] = data.get('aws_secret_key')
    payload['jsonData'] = json_data
    payload['secureJsonData'] = secure_json_data
    return payload