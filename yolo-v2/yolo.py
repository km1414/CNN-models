"""
YOLO v2 model for objects detection from image, video and live stream from webcam.

Adapted from:
    https://github.com/experiencor/basic-yolo-keras
    https://github.com/allanzelener/YAD2K

YOLO papers:
    https://arxiv.org/pdf/1506.02640.pdf
    https://arxiv.org/pdf/1612.08242.pdf

YOLO website:
    https://pjreddie.com/darknet/yolo/

Python 3. Keras. December 2017.
"""


from helpers import load_weights, filter_boxes, non_max_suppress, generate_colors, draw_boxes
from architecture import create_model, LABELS, IMAGE_H, IMAGE_W
import numpy as np
from tqdm import tqdm
import cv2


# Thresholds for confidence score and non-max suppression
OBJ_THRESHOLD = 0.4
NMS_THRESHOLD = 0.3


# Uncomment and download weighs file if needed
# download_weights()


# Create model and load weights
yolo = create_model()
load_weights(yolo, 'yolo.weights')
yolo.summary()


# All yolo actions from input to output
def make_yolo(original_image):

    input_image = cv2.resize(original_image, (IMAGE_H, IMAGE_W)) / 255.
    input_image = input_image[:, :, ::-1]
    input_image = np.expand_dims(input_image, 0)
    yolo_output = np.squeeze(yolo.predict(input_image))
    boxes = filter_boxes(yolo_output, OBJ_THRESHOLD)
    boxes = non_max_suppress(boxes, NMS_THRESHOLD)
    colours = generate_colors(LABELS)
    output_image = draw_boxes(original_image, boxes, LABELS, colours)

    return output_image


################### TEST YOLO ON IMAGE ###################


# Objects detection from image
def yolo_image(image_path):

    original_image = cv2.imread(image_path)
    image = make_yolo(original_image)
    cv2.imshow('frame', image)


yolo_image('images/test.jpg')


################### TEST YOLO ON VIDEO ###################


# Objects detection from video
def yolo_video(video_path, faster_times=1):

    # Path for output video
    video_out = '/'.join(video_path.split('/')[:-1]) + '/out_' + video_path.split('/')[-1]

    # Set video reader and writer
    video_reader = cv2.VideoCapture(video_path)
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(video_reader.get(cv2.CAP_PROP_FPS))
    video_writer = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'XVID'),
                                   fps * faster_times, (frame_w, frame_h))
    # Iterate over all frames
    for _ in tqdm(range(nb_frames)):

        ret, original_image = video_reader.read()
        image = make_yolo(original_image)
        video_writer.write(np.uint8(image))

    video_reader.release()
    video_writer.release()


yolo_video('images/test.mp4', faster_times=2)


################### YOLO LIVE STREAM ###################


# Objects detection from live stream of webcam
def yolo_live(mirror=True):

    cap = cv2.VideoCapture(0)

    # Capture frame-by-frame until quit
    while(True):

        ret, frame = cap.read()
        if mirror:
            frame = cv2.flip(frame, 1)
        image = make_yolo(frame)

        # Display the resulting frame
        cv2.imshow('frame',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


# Press 'q' to quit.
yolo_live()

