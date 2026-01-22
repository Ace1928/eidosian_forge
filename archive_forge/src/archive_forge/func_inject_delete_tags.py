from boto3.resources.action import CustomModeledAction
def inject_delete_tags(event_emitter, **kwargs):
    action_model = {'request': {'operation': 'DeleteTags', 'params': [{'target': 'Resources[0]', 'source': 'identifier', 'name': 'Id'}]}}
    action = CustomModeledAction('delete_tags', action_model, delete_tags, event_emitter)
    action.inject(**kwargs)