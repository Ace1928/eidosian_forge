#!/usr/bin/env bash
set -euo pipefail

SERVICE_DIR="${HOME}/.config/systemd/user"
UNIT_PATH="${SERVICE_DIR}/eidos-agent-link@.service"
BRIDGE_ROOT="${HOME}/.eidosian/agent_bridge"

mkdir -p "${SERVICE_DIR}" "${BRIDGE_ROOT}/logs" "${BRIDGE_ROOT}/state"

cat > "${UNIT_PATH}" <<'UNIT'
[Unit]
Description=Eidos Agent Link (%i)
After=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/env python %h/.eidosian/agent_bridge/bin/agent_link.py --bridge %h/.eidosian/agent_bridge watch --agent-id %i --ack --only-unacked --poll-interval 1.5 --heartbeat-interval 5.0
Restart=always
RestartSec=2
StandardOutput=append:%h/.eidosian/agent_bridge/logs/agent_link_%i.service.log
StandardError=append:%h/.eidosian/agent_bridge/logs/agent_link_%i.service.log

[Install]
WantedBy=default.target
UNIT

systemctl --user daemon-reload
echo "Installed ${UNIT_PATH}"
echo "Enable/start with:"
echo "  systemctl --user enable --now eidos-agent-link@<agent-id>.service"
