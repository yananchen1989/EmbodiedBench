#!/bin/bash
# 启动 X server（以无物理显示方式运行）：
# Xorg -noreset -sharevts -novtswitch -config /etc/X11/xorg.conf :0 &
# # 等待 X server 启动
# sleep 3

# # 可选：检查 OpenGL 是否正常工作，例如运行 glxinfo
# glxinfo | head -n 10

# 启动 AI2-THOR 或你的 RL 训练脚本（例如 python script.py）
# python your_ai2thor_script.py
exec bash