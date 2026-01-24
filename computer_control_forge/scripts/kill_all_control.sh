#!/bin/bash
# Kill all computer control processes
# This is the emergency stop script

echo "🛑 Engaging kill switch..."
touch /tmp/eidosian_control_kill

# Try to kill by PID if available
if [ -f /tmp/eidosian_control.pid ]; then
    PID=$(cat /tmp/eidosian_control.pid)
    echo "Stopping process $PID..."
    kill -TERM "$PID" 2>/dev/null || true
    sleep 1
    kill -9 "$PID" 2>/dev/null || true
    rm -f /tmp/eidosian_control.pid
fi

echo "✅ Kill switch engaged. Control stopped."
echo ""
echo "To resume control, run:"
echo "  rm /tmp/eidosian_control_kill"
