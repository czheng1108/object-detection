# Building Detection
## Environment Configuration
* See for details ```requirements.txt```
* Best to use Linux operating system
* The GPU model used in this project is GeForce RTX 3080

## File Structure
```
  ├── backbone: ResNet50+FPN
  ├── configs: Network Configuration Files
  ├── data_enhancement: Data enhancement
  │   ├── adding_xml: Mosaic data enhancement
  │   └── dcgan: Generative Adversarial Network
  ├── log: Used to save training and validation log files
  │   ├── loss_and_lr: Training loss and learning rate
  │   ├── mAP: mAP for each epoch
  │   ├── results_file: Training results for each epoch
  │   └── terminal_log: Terminal data
  ├── network_files: RetinaNet-SPP network
  ├── pre_training_weights: Pre-training weight storage file
  ├── save_weights: Training weight storage file
  ├── test: Save test image
  ├── test_result: Save test results
  ├── train_utils: Training and verification related modules
  ├── utils: Other configuration files
  ├── VOCdevkit: Datasets
  │   ├── VOC2012: Pascal Voc datasets
  │   ├── BCDatasets: Building under construction datasets
  │   └── pascal_voc_classes.json: PascalVoc label files 
  ├── my_dataset.py: Dataset reading script(default PascalVoc)
  ├── train.py: Training script
  ├── train_multi_GPU.py: Multi-GPU training script
  ├── predict.py: Predict script
  └── validation.py: Validation script
```

## Download pre-training weights
* ResNet50+FPN backbone: Please download the pre-training weights from [here](https://-download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth)
* RetinaNet-SPP: Please download the pre-training weights from [here](https://www.baidu.com)

## Datasets
* The format of the datasets used in this project is PascalVoc
* The download [address](http://-host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) of Pascal Voc2012 datasets
* The download [address](https://-pan.baidu.com/s/1xC-0rzzBvOBLNmJQQvQcyQ) of Buildings under Construction datasets

## Training
### Single GPU training
* Download the pre-training weights then place it to ```pre_training_weights```
* Modify the configuration file ```configs/configs```
* Enter the command in the terminal```python train.py```
### Multi GPU training
* Enter the command in the terminal```python -m torch.distributed.launch --nproc_per_node=1 --use_env train_multi_GPU.py```,
```nproc_per_node``` means the count of GPU
* If you want to specify the GPU device, please enter the command in the terminal 
  ```CUDA_VISIBLE_DEVICES=0,2 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py```

## Predict
* Place the image to be predicted in ```test``` folder
* Place the training weights in ```pre_training_weights```
* Modify the parameters in predict script ```train_weights=path```
* Enter the command in the terminal```python predict.py```
* View results in ```test_result``` folder


## Reference
* https://github.com/miaoshuyu/object-detection-usages
* https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
* https://arxiv.org/abs/1708.02002
* https://blog.csdn.net/weixin_41803339/article/details/106372080
* https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_object_detection/retinaNet
* https://github.com/pytorch/vision/tree/master/torchvision/models/detection

