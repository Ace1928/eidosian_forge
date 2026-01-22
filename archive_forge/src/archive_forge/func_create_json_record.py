from __future__ import absolute_import, division, print_function
import crypt
import json
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import jsonify
from ansible.module_utils.common.text.formatters import human_to_bytes
def create_json_record(self, create=False):
    record = {}
    user_metadata = {}
    self.result['changed'] = False
    if not create:
        rc, user_metadata, stderr = self.get_user_metadata()
        user_metadata = json.loads(user_metadata)
        user_metadata.pop('signature', None)
        user_metadata.pop('binding', None)
        user_metadata.pop('status', None)
        user_metadata.pop('lastChangeUSec', None)
        record = user_metadata
    record['userName'] = self.name
    record['secret'] = {'password': [self.password]}
    if create:
        password_hash = self._hash_password(self.password)
        record['privileged'] = {'hashedPassword': [password_hash]}
        self.result['changed'] = True
    if self.uid and self.gid and create:
        record['uid'] = self.uid
        record['gid'] = self.gid
        self.result['changed'] = True
    if self.memberof:
        member_list = list(self.memberof.split(','))
        if member_list != record.get('memberOf', [None]):
            record['memberOf'] = member_list
            self.result['changed'] = True
    if self.realname:
        if self.realname != record.get('realName'):
            record['realName'] = self.realname
            self.result['changed'] = True
    if self.storage and create:
        record['storage'] = self.storage
        self.result['changed'] = True
    if self.homedir and create:
        record['homeDirectory'] = self.homedir
        self.result['changed'] = True
    if self.imagepath and create:
        record['imagePath'] = self.imagepath
        self.result['changed'] = True
    if self.disksize:
        if self.disksize != record.get('diskSize'):
            record['diskSize'] = human_to_bytes(self.disksize)
            self.result['changed'] = True
    if self.realm:
        if self.realm != record.get('realm'):
            record['realm'] = self.realm
            self.result['changed'] = True
    if self.email:
        if self.email != record.get('emailAddress'):
            record['emailAddress'] = self.email
            self.result['changed'] = True
    if self.location:
        if self.location != record.get('location'):
            record['location'] = self.location
            self.result['changed'] = True
    if self.iconname:
        if self.iconname != record.get('iconName'):
            record['iconName'] = self.iconname
            self.result['changed'] = True
    if self.skeleton:
        if self.skeleton != record.get('skeletonDirectory'):
            record['skeletonDirectory'] = self.skeleton
            self.result['changed'] = True
    if self.shell:
        if self.shell != record.get('shell'):
            record['shell'] = self.shell
            self.result['changed'] = True
    if self.umask:
        if self.umask != record.get('umask'):
            record['umask'] = self.umask
            self.result['changed'] = True
    if self.environment:
        if self.environment != record.get('environment', [None]):
            record['environment'] = list(self.environment.split(','))
            self.result['changed'] = True
    if self.timezone:
        if self.timezone != record.get('timeZone'):
            record['timeZone'] = self.timezone
            self.result['changed'] = True
    if self.locked:
        if self.locked != record.get('locked'):
            record['locked'] = self.locked
            self.result['changed'] = True
    if self.passwordhint:
        if self.passwordhint != record.get('privileged', {}).get('passwordHint'):
            record['privileged']['passwordHint'] = self.passwordhint
            self.result['changed'] = True
    if self.sshkeys:
        if self.sshkeys != record.get('privileged', {}).get('sshAuthorizedKeys'):
            record['privileged']['sshAuthorizedKeys'] = list(self.sshkeys.split(','))
            self.result['changed'] = True
    if self.language:
        if self.locked != record.get('preferredLanguage'):
            record['preferredLanguage'] = self.language
            self.result['changed'] = True
    if self.notbefore:
        if self.locked != record.get('notBeforeUSec'):
            record['notBeforeUSec'] = self.notbefore
            self.result['changed'] = True
    if self.notafter:
        if self.locked != record.get('notAfterUSec'):
            record['notAfterUSec'] = self.notafter
            self.result['changed'] = True
    if self.mountopts:
        opts = list(self.mountopts.split(','))
        if 'nosuid' in opts:
            if record.get('mountNoSuid') is not True:
                record['mountNoSuid'] = True
                self.result['changed'] = True
        elif record.get('mountNoSuid') is not False:
            record['mountNoSuid'] = False
            self.result['changed'] = True
        if 'nodev' in opts:
            if record.get('mountNoDevices') is not True:
                record['mountNoDevices'] = True
                self.result['changed'] = True
        elif record.get('mountNoDevices') is not False:
            record['mountNoDevices'] = False
            self.result['changed'] = True
        if 'noexec' in opts:
            if record.get('mountNoExecute') is not True:
                record['mountNoExecute'] = True
                self.result['changed'] = True
        elif record.get('mountNoExecute') is not False:
            record['mountNoExecute'] = False
            self.result['changed'] = True
    return jsonify(record)