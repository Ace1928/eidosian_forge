import json
def pre_record_hook(interaction, cassette):
    """Hook to mask saved data.

    This hook will be triggered before saving the interaction, and
    will perform two tasks:
    - mask user, project and password in the saved data
    - set token expiration time to an inifinite time.
    """
    request_body = interaction.data['request']['body']
    if request_body.get('string'):
        parsed_content = json.loads(request_body['string'])
        mask_fixture_values(parsed_content, None)
        request_body['string'] = json.dumps(parsed_content)
    response_body = interaction.data['response']['body']
    if response_body.get('string'):
        parsed_content = json.loads(response_body['string'])
        mask_fixture_values(parsed_content, None)
        response_body['string'] = json.dumps(parsed_content)