from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AzureRMModuleBase
def get_webapp_ftp_publish_url(self, resource_group, name):
    self.log('Get web app {0} app publish profile'.format(name))
    url = None
    try:
        publishing_profile_options = CsmPublishingProfileOptions(format='Ftp')
        content = self.web_client.web_apps.list_publishing_profile_xml_with_secrets(resource_group_name=resource_group, name=name, publishing_profile_options=publishing_profile_options)
        if not content:
            return url
        full_xml = ''
        for f in content:
            full_xml += f.decode()
        profiles = xmltodict.parse(full_xml, xml_attribs=True)['publishData']['publishProfile']
        if not profiles:
            return url
        for profile in profiles:
            if profile['@publishMethod'] == 'FTP':
                url = profile['@publishUrl']
    except Exception as ex:
        self.fail('Error getting web app {0} app settings - {1}'.format(name, str(ex)))
    return url