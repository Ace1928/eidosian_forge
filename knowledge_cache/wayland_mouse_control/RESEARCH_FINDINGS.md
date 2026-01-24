# Wayland Mouse Control Research

## Date: 2026-01-23
## Sources Consulted:
- https://github.com/ReimuNotMoe/ydotool/issues/195
- https://unix.stackexchange.com/questions/422698/how-to-set-absolute-mouse-cursor-position-in-wayland-without-using-mouse
- KDE Discuss forums on mouse acceleration

## Key Findings

### Why Absolute Positioning Fails with ydotool
1. ydotool relies on **relative pointer emulation**
2. For absolute movement, it sends a large negative movement to (0,0) then moves relatively
3. Mouse acceleration is applied to ALL movements including virtual devices
4. Wayland security model intentionally restricts absolute mouse positioning

### Solutions

#### Solution 1: Disable Acceleration for ydotool Device via KDE D-Bus
```bash
# Find ydotool device
qdbus org.kde.KWin /org/kde/KWin/InputDevice

# For each device, check name and set flat acceleration
qdbus org.kde.KWin /org/kde/KWin/InputDevice/eventX org.kde.KWin.InputDevice.pointerAccelerationProfileFlat true
qdbus org.kde.KWin /org/kde/KWin/InputDevice/eventX org.kde.KWin.InputDevice.pointerAcceleration 0
```

#### Solution 2: Use evemu for Direct Event Injection
```bash
# Install evemu-tools
sudo apt install evemu-tools

# List devices
sudo evemu-describe

# Inject relative events
sudo evemu-event /dev/input/eventX --type EV_REL --code REL_X --value 20
sudo evemu-event /dev/input/eventX --type EV_REL --code REL_Y --value 10 --sync
```

#### Solution 3: Create Virtual Device with EV_ABS Events
A tablet-like device with absolute positioning events would bypass mouse acceleration.
Requires uinput programming.

### Current Workaround (Calibration Approach)
1. Repeatedly move to (0,0) with absolute commands
2. Use relative movement from known corner
3. Apply inverse acceleration correction (if acceleration curve is known)

### KDE-Specific Commands
```bash
# List all input devices
qdbus org.kde.KWin /org/kde/KWin/InputDevice

# Get device properties
qdbus org.kde.KWin /org/kde/KWin/InputDevice/eventX

# Set flat acceleration profile
qdbus org.kde.KWin /org/kde/KWin/InputDevice/eventX org.kde.KWin.InputDevice.pointerAccelerationProfileFlat true
```

## Recommended Implementation
1. First, find the ydotool virtual device event number
2. Set flat acceleration profile for that device
3. If that fails, use evemu for direct low-level control
4. Implement position feedback loop using screenshot + cursor detection
