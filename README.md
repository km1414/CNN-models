# CNN-models

## [YOLO-v2](/yolo-v2)


*YOLO v2 model for objects detection from image, video and live stream from webcam. 
Runs on pre-trained weights, GPU is preferred, CPU works as well.*

<img src="yolo-v2/gifs/gif_1.gif" height="160"/> <img src="yolo-v2/gifs/gif_2.gif" height="160"/> <img src="yolo-v2/gifs/gif_3.gif" height="160"/>

Papers: 
* https://arxiv.org/pdf/1506.02640.pdf
* https://arxiv.org/pdf/1612.08242.pdf 
    
Website: 
* https://pjreddie.com/darknet/yolo/ <br />

## Setup:

```
git clone https://github.com/km1414/CNN-models.git
cd CNN-models/yolo-v2
pip3 install -r requirements.txt
python3 yolo.py download_weights 
```

## Run:

Run on picture: ***python3 run_picture path/to/file***. Output will be save to ***images/***. Example:
```
python3 yolo.py run_picture images/test.jpg
```
Run on video: ***python3 run_video path/to/file***. Output will be save to ***images/***. Example:
```
python3 yolo.py run_picture images/test.mp4
```
Run live stream:
```
python3 yolo.py run_live
```


## [ResNet-32](/resnet-32)

ResNet-32 model for CIFAR-10 image recognition. <br />
Keras. Python 3. <br />
Paper: https://arxiv.org/pdf/1512.03385.pdf <br />
Some results: <br />
![](/resnet-32/ResNet-32_epochs_300_small.png) <br />


## [GoogLeNet-lite](/googlenet-lite)
GoogLeNet-lite (Inception_v1) model for CIFAR-10 image recognition. <br />
Keras. Python 3. <br />
Paper: https://arxiv.org/pdf/1409.4842.pdf <br />
Some results: <br />
![](/googlenet-lite/GoogLeNet_epochs_300_small.png) <br />

