from __future__ import annotations
import logging
import os
from typing import TYPE_CHECKING, List, Optional, Tuple
def get_gmail_credentials(token_file: Optional[str]=None, client_secrets_file: Optional[str]=None, scopes: Optional[List[str]]=None) -> Credentials:
    """Get credentials."""
    Request, Credentials = import_google()
    InstalledAppFlow = import_installed_app_flow()
    creds = None
    scopes = scopes or DEFAULT_SCOPES
    token_file = token_file or DEFAULT_CREDS_TOKEN_FILE
    client_secrets_file = client_secrets_file or DEFAULT_CLIENT_SECRETS_FILE
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, scopes)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(client_secrets_file, scopes)
            creds = flow.run_local_server(port=0)
        with open(token_file, 'w') as token:
            token.write(creds.to_json())
    return creds