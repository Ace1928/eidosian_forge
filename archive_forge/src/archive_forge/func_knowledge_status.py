from __future__ import annotations
from typing import Any, Dict, Optional
import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
def knowledge_status(self, document_name: str) -> dict:
    """
        Use this function to know the status of the document or the URL uploaded
        Args:
            document_name (str): The document name or the url that is uploaded.

        Returns:
            dict: Response JSON from the Cogniswitch service.
        """
    params = {'docName': document_name, 'platformToken': self.cs_token}
    headers = {'apiKey': self.apiKey, 'openAIToken': self.OAI_token, 'platformToken': self.cs_token}
    response = requests.get(self.knowledge_status_url, headers=headers, params=params, verify=False)
    if response.status_code == 200:
        source_info = response.json()
        source_data = dict(source_info[-1])
        status = source_data.get('status')
        if status == 0:
            source_data['status'] = 'SUCCESS'
        elif status == 1:
            source_data['status'] = 'PROCESSING'
        elif status == 2:
            source_data['status'] = 'UPLOADED'
        elif status == 3:
            source_data['status'] = 'FAILURE'
        elif status == 4:
            source_data['status'] = 'UPLOAD_FAILURE'
        elif status == 5:
            source_data['status'] = 'REJECTED'
        if 'filePath' in source_data.keys():
            source_data.pop('filePath')
        if 'savedFileName' in source_data.keys():
            source_data.pop('savedFileName')
        if 'integrationConfigId' in source_data.keys():
            source_data.pop('integrationConfigId')
        if 'metaData' in source_data.keys():
            source_data.pop('metaData')
        if 'docEntryId' in source_data.keys():
            source_data.pop('docEntryId')
        return source_data
    else:
        return {'message': response.status_code}