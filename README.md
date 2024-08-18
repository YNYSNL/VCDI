# VUT-Centered environmental Dynamics Inference
This repo is the implementation of the following paper:
<<<<<<< HEAD

**Towards Interactive Autonomous Vehicle Testing: Vehicle-Under-Test-Centered Traffic Simulation**
<br> [Yiru Liu](https://github.com/YNYSNL), [Xiaocong Zhao](https://zxc-tju.github.io), [Jian Sun](https://scholar.google.com/citations?hl=zh-CN&user=dXaFOeYAAAAJ) 
<br> **[[arXiv]](https://arxiv.org/abs/2406.02860)**

## Dataset
Download the [Waymo Open Motion Dataset](https://waymo.com/open/download/) v1.1; only the files in ```uncompressed/scenario/training_20s``` are needed. Place the downloaded files into training and testing folders separately.

## Installation
### Install dependency
```bash
sudo apt-get install libsuitesparse-dev
```

### Create conda env
```bash
conda env create -f environment.yml
conda activate VCDI
```

### Install Theseus
Install cuda 11.3 in advance.
```bash
pip install functorch
pip install theseus-ai
```

## Usage
### Data Processing
Run ```data_process.py``` to process the raw data for training. This will convert the original data format into a set of ```.npz``` files, each containing the data of a scene with the AV and surrounding agents. You need to specify the file path to the original data ```--load_path``` and the path to save the processed data ```--save_path``` . You can optionally set ```--use_multiprocessing``` to speed up the processing. 
```shell
python data_process.py \
--load_path /path/to/original/data \
--save_path /output/path/to/processed/data \
--use_multiprocessing
```

### Training
Run ```train.py``` to learn the predictor and planner (if set ```--use_planning```). You need to specify the file paths to training data ```--train_set```. Leave other arguments vacant to use the default setting. You can choose different model types ```--model_type``` to train the three models mentioned in the paper.
```shell
python train.py \
--name VCDI \
--train_set /path/to/train/data \
--pretrain_epochs 5 \
--train_epochs 30 \
--batch_size 32 \
--learning_rate 2e-4 \
--use_planning \
--model_type VCDI \
--device cuda
```

### Validating
Run ```test_model.py``` to evaluate the performance of the three models you trained with two key metrics: the average displacement error (ADE) and the final displacement error (FDE) over three timesteps (1, 3, and 5 seconds). You need to specify the path to the test dataset ```--test_set``` and also the file path to the trained model ```--model_path```.
```shell
python test_model.py \
--name VCDI \
--model_path /path/to/saved/model \
--test_set /path/to/test/data \
--batch_size 32 \
--use_planning \
--model_type VCDI \
--device cuda
```

### Open-loop testing
Run ```open_loop_test.py``` to test the trained planner in an open-loop manner. You need to specify the path to the original test dataset ```--open_loop_test_set``` (path to the folder) and also the file path to the trained model ```--model_path```. Set ```--render``` to visualize the results and set ```--save``` to save the rendered images.
```shell
python open_loop_test.py \
--name open_loop \
--open_loop_test_set /path/to/original/test/data \
--model_path /path/to/saved/model \
--use_planning \
--model_type VCDI \
--render \
--save \
--device cpu
```

## Citation
If you find our repo or our paper useful, please use the following citation:
```
@article{liu2024towards,
  title={Towards Interactive Autonomous Vehicle Testing: Vehicle-Under-Test-Centered Traffic Simulation},
  author={Liu, Yiru and Zhao, Xiaocong and Sun, Jian},
  journal={arXiv preprint arXiv:2406.02860},
  year={2024}
}
```
## Acknowledgment
This project is a fork of [DIPP](https://github.com/MCZhi/DIPP) by [Zhiyu Huang](https://mczhi.github.io/) et al. I would like to thank the original authors for their work and contribution to the community.
