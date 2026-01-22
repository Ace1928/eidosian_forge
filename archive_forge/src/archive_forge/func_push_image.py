from __future__ import absolute_import, division, print_function
import json
import re
import shlex
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.containers.podman.plugins.module_utils.podman.common import run_podman_command
def push_image(self):
    args = ['push']
    if self.validate_certs is not None:
        if self.validate_certs:
            args.append('--tls-verify')
        else:
            args.append('--tls-verify=false')
    if self.ca_cert_dir:
        args.extend(['--cert-dir', self.ca_cert_dir])
    if self.username and self.password:
        cred_string = '{user}:{password}'.format(user=self.username, password=self.password)
        args.extend(['--creds', cred_string])
    if self.auth_file:
        args.extend(['--authfile', self.auth_file])
    if self.push_args.get('compress'):
        args.append('--compress')
    push_format = self.push_args.get('format')
    if push_format:
        args.extend(['--format', push_format])
    if self.push_args.get('remove_signatures'):
        args.append('--remove-signatures')
    sign_by_key = self.push_args.get('sign_by')
    if sign_by_key:
        args.extend(['--sign-by', sign_by_key])
    args.append(self.image_name)
    dest = self.push_args.get('dest')
    dest_format_string = '{dest}/{image_name}'
    regexp = re.compile('/{name}(:{tag})?'.format(name=self.name, tag=self.tag))
    if not dest:
        if '/' not in self.name:
            self.module.fail_json(msg="'push_args['dest']' is required when pushing images that do not have the remote registry in the image name")
    elif regexp.search(dest):
        dest = regexp.sub('', dest)
        self.module.warn("Image name and tag are automatically added to push_args['dest']. Destination changed to {dest}".format(dest=dest))
    if dest and dest.endswith('/'):
        dest = dest[:-1]
    transport = self.push_args.get('transport')
    if transport:
        if not dest:
            self.module.fail_json("'push_args['transport'] requires 'push_args['dest'] but it was not provided.")
        if transport == 'docker':
            dest_format_string = '{transport}://{dest}'
        elif transport == 'ostree':
            dest_format_string = '{transport}:{name}@{dest}'
        else:
            dest_format_string = '{transport}:{dest}'
    dest_string = dest_format_string.format(transport=transport, name=self.name, dest=dest, image_name=self.image_name)
    if '/' not in self.name:
        args.append(dest_string)
    rc, out, err = self._run(args, ignore_errors=True)
    if rc != 0:
        self.module.fail_json(msg='Failed to push image {image_name}: {err}'.format(image_name=self.image_name, err=err))
    last_id = self._get_id_from_output(out + err, contains=':', split_on=':')
    return (self.inspect_image(last_id), out + err)