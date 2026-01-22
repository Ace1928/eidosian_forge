import logging
import time
from queue import Queue
import uuid
from parlai.core.agents import Agent
import parlai.mturk.core.data_model as data_model
import parlai.mturk.core.shared_utils as shared_utils
def get_inactive_command_text(self):
    """
        Get appropriate inactive command and text to respond to a reconnect given the
        current assignment state.

        returns text, command
        """
    command = data_model.COMMAND_INACTIVE_HIT
    text = None
    if self.status == self.STATUS_DISCONNECT:
        text = 'You disconnected in the middle of this HIT and were marked as inactive. As these HITs often require real-time interaction, it is no longer available for completion. Please return this HIT and accept a new one if you would like to try again.'
    elif self.status == self.STATUS_DONE:
        command = data_model.COMMAND_INACTIVE_DONE
        text = 'You disconnected after completing this HIT without marking it as completed. Please press the done button below to finish the HIT.'
    elif self.status == self.STATUS_EXPIRED:
        text = 'You disconnected in the middle of this HIT and the HIT expired before you reconnected. It is no longer available for completion. Please return this HIT and accept a new one if you would like to try again.'
    elif self.status == self.STATUS_PARTNER_DISCONNECT:
        command = data_model.COMMAND_INACTIVE_DONE
        text = "One of your partners disconnected in the middle of the HIT. We won't penalize you for their disconnect, so please use the button below to mark the HIT as complete."
    elif self.status == self.STATUS_PARTNER_DISCONNECT_EARLY:
        command = data_model.COMMAND_INACTIVE_HIT
        text = "One of your partners disconnected in the middle of the HIT. We won't penalize you for their disconnect, but you did not complete enough of the task to submit the HIT. Please return this HIT and accept a new one if you would like to try again."
    elif self.status == self.STATUS_RETURNED:
        text = 'You disconnected from this HIT and then returned it. As we have marked the HIT as returned, it is no longer available for completion. Please accept a new HIT if you would like to try again'
    else:
        text = 'Our server was unable to handle your reconnect properly and thus this HIT no longer seems available for completion. Please try to connect again or return this HIT and accept a new one.'
    return (text, command)