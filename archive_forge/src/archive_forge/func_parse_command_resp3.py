import datetime
from redis.utils import str_if_bytes
def parse_command_resp3(response, **options):
    commands = {}
    for command in response:
        cmd_dict = {}
        cmd_name = str_if_bytes(command[0])
        cmd_dict['name'] = cmd_name
        cmd_dict['arity'] = command[1]
        cmd_dict['flags'] = {str_if_bytes(flag) for flag in command[2]}
        cmd_dict['first_key_pos'] = command[3]
        cmd_dict['last_key_pos'] = command[4]
        cmd_dict['step_count'] = command[5]
        cmd_dict['acl_categories'] = command[6]
        if len(command) > 7:
            cmd_dict['tips'] = command[7]
            cmd_dict['key_specifications'] = command[8]
            cmd_dict['subcommands'] = command[9]
        commands[cmd_name] = cmd_dict
    return commands