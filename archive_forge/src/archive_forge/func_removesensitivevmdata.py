from __future__ import (absolute_import, division, print_function)
from xml.etree import ElementTree
def removesensitivevmdata(self, data, key_to_remove='root_password'):
    for value in data:
        if key_to_remove in value:
            value[key_to_remove] = '******'
        if 'cloud_init' in value and key_to_remove in value['cloud_init']:
            value['cloud_init'][key_to_remove] = '******'
        if 'sysprep' in value and key_to_remove in value['sysprep']:
            value['sysprep'][key_to_remove] = '******'
        if 'profile' in value:
            profile = value['profile']
            if key_to_remove in profile:
                profile[key_to_remove] = '******'
            if 'cloud_init' in profile and key_to_remove in profile['cloud_init']:
                profile['cloud_init'][key_to_remove] = '******'
            if 'sysprep' in profile and key_to_remove in profile['sysprep']:
                profile['sysprep'][key_to_remove] = '******'
    return data