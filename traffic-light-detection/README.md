# Traffic Light Detection

To detect traffic lights we've decided to use [TensorFlow Object Detection API](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI) 

## Setup

1. Create an AWS spot instance from the AMI `udacity-carnd-advanced-deep-learning`
2. Run `ssh -i your-pem-here.pem ubuntu@aws-public-dns-here`
3. Run `sudo apt-get update`
4. Run `git clone https://github.com/ricardosllm/CarND-Capstone.git && cd CarND-Capstone`
5. Run `git checkout deep-traffic-light && cd traffic-light-detection`
6. Run `make setup`
7. Run `source ~/.bashrc`
8. Run `cd ~/tensorflow/models/research`
9. Run one of the following based on the model you chose and sim/real images

## Using Inception SSD v2
NOTE: To use Inception SSD v2, you must update the path to the model on line 152. Sim [here](https://github.com/ricardosllm/TrafficLight_Detection-TensorFlowAPI/blob/master/config/ssd_inception-traffic-udacity_sim.config) and Real [here](https://github.com/ricardosllm/TrafficLight_Detection-TensorFlowAPI/blob/master/config/ssd_inception-traffic_udacity_real.config)

Change it to: `ssd_inception_v2_coco_2017_11_17`

### For Simulator Data

#### Training

`python object_detection/train.py --pipeline_config_path=config/ssd_inception-traffic-udacity_sim.config --train_dir=data/sim_training_data/sim_data_capture`

#### Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/ssd_inception-traffic-udacity_sim.config --trained_checkpoint_prefix=data/sim_training_data/sim_data_capture/model.ckpt-5000 --output_directory=frozen_models/frozen_sim_inception/`


### For Real Data

#### Training

`python object_detection/train.py --pipeline_config_path=config/ssd_inception-traffic_udacity_real.config --train_dir=data/real_training_data`

#### Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/ssd_inception-traffic_udacity_real.config --trained_checkpoint_prefix=data/real_training_data/model.ckpt-10000 --output_directory=frozen_models/frozen_real_inception/`

## Using Faster-RCNN model

### For Simulator Data

#### Training

`python object_detection/train.py --pipeline_config_path=config/faster_rcnn-traffic-udacity_sim.config --train_dir=data/sim_training_data/sim_data_capture`

#### Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/faster_rcnn-traffic-udacity_sim.config --trained_checkpoint_prefix=data/sim_training_data/sim_data_capture/model.ckpt-5000 --output_directory=frozen_sim/`


### For Real Data

#### Training

`python object_detection/train.py --pipeline_config_path=config/faster_rcnn-traffic_udacity_real.config --train_dir=data/real_training_data`

#### Saving for Inference

`python object_detection/export_inference_graph.py --pipeline_config_path=config/faster_rcnn-traffic_udacity_real.config --trained_checkpoint_prefix=data/real_training_data/model.ckpt-10000 --output_directory=frozen_real/`




