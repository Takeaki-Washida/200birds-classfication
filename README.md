# 200birds-classfication
卒業研究　graduation research

===
##Requirement
# Used module
* python3
* keras 2.2.4
* opencv-python 3.4.3.18
* Pillow 5.3.0
* tensorflow-gpu 1.12.0
* numpy 1.15.4

#finetuning.py
* training bird dataset
python3 finetuning.py

#grad_cam_segmentation.py
提案手法１

#grad_cam_outbird.py
提案手法２

#model.py
define deeplabv3+ model

##birds dataset
http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

## Usage
** RUN  training program
 '$ python3 finetuning.py

** RUN classfication program
 '$ python3 grad_cam_segmentation.py ディレクトリパス
 '$ python3 grad_cam_outbird.py ディレクトリパス
