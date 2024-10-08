# bjj dataset of images with keypoints annotations in MS-COCO format
https://vicos.si/resources/jiujitsu/

# Proper way to install mmcv by selecting platform/cuda-cpu/torch-version/mmcv-version
https://mmcv.readthedocs.io/en/latest/get_started/installation.html#install-with-pip

python -m ensurepip --upgrade
python -m pip install --upgrade setuptools

pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"

conda install scikit-learn

# contains Deformable-Detr (object detection)
pip install transformers