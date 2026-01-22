from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def construct_curated_webapp(self, webapp, configuration=None, app_settings=None, deployment_slot=None, ftp_publish_url=None, publish_credentials=None):
    curated_output = dict()
    curated_output['id'] = webapp['id']
    curated_output['name'] = webapp['name']
    curated_output['resource_group'] = webapp['resource_group']
    curated_output['location'] = webapp['location']
    curated_output['plan'] = webapp['server_farm_id']
    curated_output['tags'] = webapp.get('tags', None)
    curated_output['app_state'] = webapp['state']
    curated_output['availability_state'] = webapp['availability_state']
    curated_output['default_host_name'] = webapp['default_host_name']
    curated_output['host_names'] = webapp['host_names']
    curated_output['enabled'] = webapp['enabled']
    curated_output['enabled_host_names'] = webapp['enabled_host_names']
    curated_output['host_name_ssl_states'] = webapp['host_name_ssl_states']
    curated_output['outbound_ip_addresses'] = webapp['outbound_ip_addresses']
    if configuration:
        curated_output['frameworks'] = []
        for fx_name in self.framework_names:
            fx_version = configuration.get(fx_name + '_version', None)
            if fx_version:
                fx = {'name': fx_name, 'version': fx_version}
                if fx_name == 'java':
                    if configuration['java_container'] and configuration['java_container_version']:
                        settings = {'java_container': configuration['java_container'].lower(), 'java_container_version': configuration['java_container_version']}
                        fx['settings'] = settings
                curated_output['frameworks'].append(fx)
        if configuration.get('linux_fx_version', None):
            tmp = configuration.get('linux_fx_version').split('|')
            if len(tmp) == 2:
                curated_output['frameworks'].append({'name': tmp[0].lower(), 'version': tmp[1]})
        curated_output['always_on'] = configuration.get('always_on')
        curated_output['ftps_state'] = configuration.get('ftps_state')
        curated_output['min_tls_version'] = configuration.get('min_tls_version')
    if app_settings and app_settings.get('properties', None):
        curated_output['app_settings'] = dict()
        for item in app_settings['properties']:
            curated_output['app_settings'][item] = app_settings['properties'][item]
    if deployment_slot:
        curated_output['deployment_slot'] = deployment_slot
    if ftp_publish_url:
        curated_output['ftp_publish_url'] = ftp_publish_url
    if publish_credentials and self.return_publish_profile:
        curated_output['publishing_username'] = publish_credentials.publishing_user_name
        curated_output['publishing_password'] = publish_credentials.publishing_password
    return curated_output