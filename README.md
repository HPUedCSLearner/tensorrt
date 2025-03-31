

## env


```
bash
docker run -it --rm \
	--network host \
	--ipc host \
	-v `pwd`:/ws \
	-w /ws \
	--gpus all \
	nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3 \
	bash


```
