from __future__ import (absolute_import, division, print_function)
from copy import deepcopy
import json
import re
import base64
import time
def get_working_environment_detail_for_snapmirror(self, rest_api, headers):
    source_working_env_detail, dest_working_env_detail = ({}, {})
    if self.parameters.get('source_working_environment_id'):
        api = '/occm/api/working-environments'
        working_env_details, error, dummy = rest_api.get(api, None, header=headers)
        if error:
            return (None, None, 'Error getting WE info: %s: %s' % (error, working_env_details))
        for dummy, values in working_env_details.items():
            for each in values:
                if each['publicId'] == self.parameters['source_working_environment_id']:
                    source_working_env_detail = each
                    break
    elif self.parameters.get('source_working_environment_name'):
        source_working_env_detail, error = self.get_working_environment_details_by_name(rest_api, headers, self.parameters['source_working_environment_name'])
        if error:
            return (None, None, error)
    else:
        return (None, None, 'Cannot find working environment by source_working_environment_id or source_working_environment_name')
    if self.parameters.get('destination_working_environment_id'):
        if self.parameters['destination_working_environment_id'].startswith('fs-'):
            if self.parameters.get('tenant_id'):
                working_env_details, error = self.get_aws_fsx_details_by_id(rest_api, header=headers)
                if error:
                    return (None, None, 'Error getting WE info for FSx: %s: %s' % (error, working_env_details))
                dest_working_env_detail['publicId'] = self.parameters['destination_working_environment_id']
                svm_name, error = self.get_aws_fsx_svm(rest_api, self.parameters['destination_working_environment_id'], header=headers)
                if error:
                    return (None, None, 'Error getting svm name for FSx: %s' % error)
                dest_working_env_detail['svmName'] = svm_name
            else:
                return (None, None, 'Cannot find FSx WE by destination WE %s, missing tenant_id' % self.parameters['destination_working_environment_id'])
        else:
            api = '/occm/api/working-environments'
            working_env_details, error, dummy = rest_api.get(api, None, header=headers)
            if error:
                return (None, None, 'Error getting WE info: %s: %s' % (error, working_env_details))
            for dummy, values in working_env_details.items():
                for each in values:
                    if each['publicId'] == self.parameters['destination_working_environment_id']:
                        dest_working_env_detail = each
                        break
    elif self.parameters.get('destination_working_environment_name'):
        if self.parameters.get('tenant_id'):
            fsx_id, error = self.get_aws_fsx_details_by_name(rest_api, header=headers)
            if error:
                return (None, None, 'Error getting WE info for FSx: %s' % error)
            dest_working_env_detail['publicId'] = fsx_id
            svm_name, error = self.get_aws_fsx_svm(rest_api, fsx_id, header=headers)
            if error:
                return (None, None, 'Error getting svm name for FSx: %s' % error)
            dest_working_env_detail['svmName'] = svm_name
        else:
            dest_working_env_detail, error = self.get_working_environment_details_by_name(rest_api, headers, self.parameters['destination_working_environment_name'])
            if error:
                return (None, None, error)
    else:
        return (None, None, 'Cannot find working environment by destination_working_environment_id or destination_working_environment_name')
    return (source_working_env_detail, dest_working_env_detail, None)