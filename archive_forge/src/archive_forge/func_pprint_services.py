import argparse
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union, cast
from langsmith import env as ls_env
from langsmith import utils as ls_utils
def pprint_services(services_status: List[Mapping[str, Union[str, List[str]]]]) -> None:
    services = []
    for service in services_status:
        service_status: Dict[str, str] = {'Service': str(service['Service']), 'Status': str(service['Status'])}
        publishers = cast(List[Dict], service.get('Publishers', []))
        if publishers:
            service_status['PublishedPorts'] = ', '.join([str(publisher['PublishedPort']) for publisher in publishers])
        services.append(service_status)
    max_service_len = max((len(service['Service']) for service in services))
    max_state_len = max((len(service['Status']) for service in services))
    service_message = ['\n' + 'Service'.ljust(max_service_len + 2) + 'Status'.ljust(max_state_len + 2) + 'Published Ports']
    for service in services:
        service_str = service['Service'].ljust(max_service_len + 2)
        state_str = service['Status'].ljust(max_state_len + 2)
        ports_str = service.get('PublishedPorts', '')
        service_message.append(service_str + state_str + ports_str)
    service_message.append('\nTo connect, set the following environment variables in your LangChain application:\nLANGSMITH_TRACING_V2=true\nLANGSMITH_ENDPOINT=http://localhost:80/api')
    logger.info('\n'.join(service_message))