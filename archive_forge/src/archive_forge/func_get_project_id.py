from the current environment without the need to copy, save and manage
import abc
import copy
import datetime
import io
import json
import re
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import impersonated_credentials
from google.auth import metrics
from google.oauth2 import sts
from google.oauth2 import utils
def get_project_id(self, request):
    """Retrieves the project ID corresponding to the workload identity or workforce pool.
        For workforce pool credentials, it returns the project ID corresponding to
        the workforce_pool_user_project.

        When not determinable, None is returned.

        This is introduced to support the current pattern of using the Auth library:

            credentials, project_id = google.auth.default()

        The resource may not have permission (resourcemanager.projects.get) to
        call this API or the required scopes may not be selected:
        https://cloud.google.com/resource-manager/reference/rest/v1/projects/get#authorization-scopes

        Args:
            request (google.auth.transport.Request): A callable used to make
                HTTP requests.
        Returns:
            Optional[str]: The project ID corresponding to the workload identity pool
                or workforce pool if determinable.
        """
    if self._project_id:
        return self._project_id
    scopes = self._scopes if self._scopes is not None else self._default_scopes
    project_number = self.project_number or self._workforce_pool_user_project
    if project_number and scopes:
        headers = {}
        url = _CLOUD_RESOURCE_MANAGER + project_number
        self.before_request(request, 'GET', url, headers)
        response = request(url=url, method='GET', headers=headers)
        response_body = response.data.decode('utf-8') if hasattr(response.data, 'decode') else response.data
        response_data = json.loads(response_body)
        if response.status == 200:
            self._project_id = response_data.get('projectId')
            return self._project_id
    return None