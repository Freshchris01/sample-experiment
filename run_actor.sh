#!/usr/bin/env

nohup python actor.py \
	--inference-mode=local \
	--env TouchCubeVector \
	--server 192.168.3.213 \
	--port 9900 \
&