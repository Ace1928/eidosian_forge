import io
import json
import logging
import mimetypes
import os
import re
import threading
import time
import urllib
import urllib.parse
from collections import deque
from datetime import datetime, timezone
from io import IOBase
from typing import (
import requests
import requests.adapters
from urllib3 import Retry
import github.Consts as Consts
import github.GithubException as GithubException
def graphql_named_mutation(self, mutation_name: str, variables: Dict[str, Any], output: Optional[str]=None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
        Create a mutation in the format:
            mutation MutationName($input: MutationNameInput!) {
                mutationName(input: $input) {
                    <output>
                }
            }
        and call the self.graphql_query method
        """
    title = ''.join([x.capitalize() for x in mutation_name.split('_')])
    mutation_name = title[:1].lower() + title[1:]
    output = output or ''
    query = f'mutation {title}($input: {title}Input!) {{ {mutation_name}(input: $input) {{ {output} }} }}'
    return self.graphql_query(query, variables)