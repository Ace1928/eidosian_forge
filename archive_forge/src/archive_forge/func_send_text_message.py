import langchain
from langchain.llms import Replicate
from flask import Flask
from flask import request
import os
import requests
import json
def send_text_message(self, message, phone_number):
    payload = {'messaging_product': 'whatsapp', 'to': phone_number, 'type': 'text', 'text': {'preview_url': False, 'body': message}}
    response = requests.post(f'{self.API_URL}/messages', json=payload, headers=self.headers)
    print(response.status_code)
    assert response.status_code == 200, 'Error sending message'
    return response.status_code