import json
import os
import numpy as np
def minerec_to_minerl_action(minerec_action, next_action=None, gui_camera_scaler=1.0, esc_to_inventory=True):
    """
    Convert minerec action into minerl action.
    :param minerec_action: action in minerec format
    :param next_action: next action (optional). If not None, camera is inferred by difference
                        between states instead of from action directly.
    :param gui_camera_scaler: additional factor to be applied to camera when gui is open. Useful
                               when replaying actions recorded with old (<=5.8 versions) of minerec
                               recorder (should be set to 0.5)
    :param esc_to_inventory: if True, map ESC key presses to inventory (e) when gui is open.
    :returns action in minerl format
    """
    ac = {v: 0 for v in KEYMAP.values()}
    ac['camera'] = np.zeros(2)
    if minerec_action.get('mouse') is None or minerec_action.get('keyboard') is None or (next_action is not None and next_action.get('mouse') is None):
        return ac
    keys_pressed = set()
    keys_pressed.update(minerec_action['keyboard']['keys'])
    keys_pressed.update((f'mouse.{b}' for b in minerec_action['mouse']['buttons']))
    ac['camera'] = mouse_to_camera(minerec_action['mouse'])
    for key, keyac in KEYMAP.items():
        if keyac == 'ESC' and minerec_action['isGuiOpen'] and esc_to_inventory and (key in keys_pressed):
            keyac = 'inventory'
        ac[keyac] = int(key in keys_pressed)
    if minerec_action['isGuiOpen']:
        ac['camera'] *= gui_camera_scaler
    if next_action is not None:
        if not minerec_action['isGuiOpen']:
            dpitch = next_action['pitch'] - minerec_action['pitch']
            dyaw = next_action['yaw'] - minerec_action['yaw']
            ac['camera'] = np.array([dpitch, dyaw])
        elif next_action['isGuiOpen']:
            if 'scaledX' in next_action['mouse']:
                dpitch = (next_action['mouse']['scaledY'] - minerec_action['mouse']['scaledY']) * CAMERA_SCALER
                dyaw = (next_action['mouse']['scaledX'] - minerec_action['mouse']['scaledX']) * CAMERA_SCALER
            else:
                dpitch = (next_action['mouse']['y'] - minerec_action['mouse']['y']) / 2 * CAMERA_SCALER
                dyaw = (next_action['mouse']['x'] - minerec_action['mouse']['x']) / 2 * CAMERA_SCALER
            ac['camera'] = np.array([dpitch, dyaw])
        if 'hotbar' in next_action:
            if next_action['hotbar'] != minerec_action['hotbar']:
                ac[f'hotbar.{next_action['hotbar'] + 1}'] = 1
        else:
            ac['dwheel'] = minerec_action['mouse']['dwheel']
    return ac