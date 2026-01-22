from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
The condition(s) under which the action will be taken.

        Fields:
          age: Age of an object (in days). This condition is satisfied when an
            object reaches the specified age.
          createdBefore: A date in RFC 3339 format with only the date part
            (for instance, "2013-01-15"). This condition is satisfied when an
            object is created before midnight of the specified date in UTC.
          customTimeBefore: A date in RFC 3339 format with only the date part
            (for instance, "2013-01-15"). This condition is satisfied when the
            custom time on an object is before this date in UTC.
          daysSinceCustomTime: Number of days elapsed since the user-specified
            timestamp set on an object. The condition is satisfied if the days
            elapsed is at least this number. If no custom timestamp is
            specified on an object, the condition does not apply.
          daysSinceNoncurrentTime: Number of days elapsed since the noncurrent
            timestamp of an object. The condition is satisfied if the days
            elapsed is at least this number. This condition is relevant only
            for versioned objects. The value of the field must be a
            nonnegative integer. If it's zero, the object version will become
            eligible for Lifecycle action as soon as it becomes noncurrent.
          isLive: Relevant only for versioned objects. If the value is true,
            this condition matches live objects; if the value is false, it
            matches archived objects.
          matchesStorageClass: Objects having any of the storage classes
            specified by this condition will be matched. Values include
            MULTI_REGIONAL, REGIONAL, NEARLINE, COLDLINE, ARCHIVE, STANDARD,
            and DURABLE_REDUCED_AVAILABILITY.
          noncurrentTimeBefore: A date in RFC 3339 format with only the date
            part (for instance, "2013-01-15"). This condition is satisfied
            when the noncurrent time on an object is before this date in UTC.
            This condition is relevant only for versioned objects.
          numNewerVersions: Relevant only for versioned objects. If the value
            is N, this condition is satisfied when there are at least N
            versions (including the live version) newer than this version of
            the object.
        