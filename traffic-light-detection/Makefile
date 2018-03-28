
RESEARCH=~/tensorflow/models/research
OBJ=${RESEARCH}/object_detection/
UTILITIES=${RESEARCH}/utilities
TLD-TAPI=TrafficLight_Detection-TensorFlowAPI
TF_MODELS_140=1464a721fef7da67e87b0f1f1a710b1e904e8764
OBJECT_DETECTION_MODEL=faster_rcnn_resnet101_coco_11_06_2017

deps:
	sudo apt-get install protobuf-compiler python-pil python-lxml python-tk

pip:
	pip install -r requirements.txt

dataset:
	sh gdrive.sh 0B-Eiyn-CUQtxdUZWMkFfQzdObUE ${RESEARCH}/data.zip && \
	cd ${RESEARCH} && unzip -o data.zip

models:
	cd ~/tensorflow && \
	git clone https://github.com/tensorflow/models.git || (echo "models already downloaded") && \
	cd ~/tensorflow/models && git checkout ${TF_MODELS_140}

cocoapi: 
	git clone https://github.com/cocodataset/cocoapi.git && \
	cd cocoapi/PythonAPI && \
	make && \
	cp -r pycocotools ${RESEARCH}

utils:
	cd ${RESEARCH} && mkdir -p utilities && \
	cd ${OBJ} && cp -r utils/* ${UTILITIES}

protobuf:
	cd  ${RESEARCH} && \
	protoc object_detection/protos/*.proto --python_out=.

pythonpath:
	echo "export PYTHONPATH=$$PYTHONPATH:${RESEARCH}:${RESEARCH}/slim" >> ~/.bashrc

pre-trained-model:
	cd ${RESEARCH} && \
	curl http://download.tensorflow.org/models/object_detection/${OBJECT_DETECTION_MODEL}.tar.gz -o ${OBJECT_DETECTION_MODEL}.tar.gz && \
	tar -xvzf ${OBJECT_DETECTION_MODEL}.tar.gz && \
	git clone https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI.git ${TLD-TAPI} || (cd ${TLD-TAPI}; git pull) && \
	cp -r ${TLD-TAPI}/* .

data-conversion:
	cd ${RESEARCH} && \
	python data_conversion_udacity_sim.py --output_path sim_data.record && \
	python data_conversion_udacity_real.py --output_path real_data.record

test-installation:
	cd ${RESEARCH} && \
	python object_detection/builders/model_builder_test.py

setup: deps pip models cocoapi protobuf pythonpath pre-trained-model dataset utils data-conversion test-installation

sim-train:
	cd ${RESEARCH} && \
	python object_detection/train.py \
	--pipeline_config_path=config/faster_rcnn-traffic-udacity_sim.config \
	--train_dir=data/sim_training_data/sim_data_capture

sim-export:
	cd ${RESEARCH} && \
	python object_detection/export_inference_graph.py \
	--pipeline_config_path=config/faster_rcnn-traffic-udacity_sim.config \
	--trained_checkpoint_prefix=data/sim_training_data/sim_data_capture/model.ckpt-5000 \
	--output_directory=frozen_sim/

