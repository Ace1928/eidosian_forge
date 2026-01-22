from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader

        init method
        :param query: sql query to execute
        :param user: username
        :param password: user password
        :param schema: schema to run in database
        :param tns_name: tns name in tnsname.ora
        :param config_dir: directory of config files(tnsname.ora, wallet)
        :param wallet_location: location of wallet
        :param wallet_password: password of wallet
        :param connection_string: connection string to connect to adb instance
        :param metadata: metadata used in document
        