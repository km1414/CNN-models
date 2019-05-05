# YOLO v2


YOLO v2 model for objects detection from image, video and live stream from webcam. 
Runs on pre-trained weights, GPU is preferred, CPU works as well.

<img src="gifs/gif_1.gif" height="100"/><img src="gifs/gif_2.gif" height="100"/><img src="gifs/gif_2.gif" height="100"/>

Papers: 
* https://arxiv.org/pdf/1506.02640.pdf
* https://arxiv.org/pdf/1612.08242.pdf 
    
Website: 
* https://pjreddie.com/darknet/yolo/ <br />

### Setup:

```
git clone https://github.com/km1414/CNN-models.git
cd CNN-models/yolo-v2
pip3 install -r requirements.txt
python3 yolo.py download_weights 
```

### Run:

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