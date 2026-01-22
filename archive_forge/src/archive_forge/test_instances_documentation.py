from novaclient.tests.functional import base
Test we can attach a volume via the cli.

        This test was added after bug 1423695. That bug exposed
        inconsistencies in how to talk to API services from the CLI
        vs. API level. The volumes api calls that were designed to
        populate the completion cache were incorrectly routed to the
        Nova endpoint. Novaclient volumes support actually talks to
        Cinder endpoint directly.

        This would case volume-attach to return a bad error code,
        however it does this *after* the attach command is correctly
        dispatched. So the volume-attach still works, but the user is
        presented a 404 error.

        This test ensures we can do a through path test of: boot,
        create volume, attach volume, detach volume, delete volume,
        destroy.

        