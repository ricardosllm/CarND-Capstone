

deps:
	sudo apt-get install protobuf-compiler python-pil python-lxml python-tk

pip:
	pip install -r requirements.txt

models:
	cd ~/tensorflow && \
	git clone https://github.com/tensorflow/models.git || (cd models; git pull); \

cocoapi: 
	git clone https://github.com/cocodataset/cocoapi.git && \
	cd cocoapi/PythonAPI && \
	make && \
	cp -r pycocotools ~/tensorflow/models/research/

protobuf:
	cd ~/tensorflow/models/research && \
	protoc object_detection/protos/*.proto --python_out=.

pythonpath:
	export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

setup: deps pip models cocoapi protobuf pythonpath
