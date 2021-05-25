import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.config import cfg
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from core.plate_recognition import recognize_plate


def recognise(fp):

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.temp_load_config(False)
    input_size = 416


    # load model

    saved_model_loaded = tf.saved_model.load('./checkpoints/custom-416', tags=[tag_constants.SERVING])

    # loop through images in list and run Yolov4 model on each

    original_image = cv2.imread(fp)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    image_data = cv2.resize(original_image, (input_size, input_size))
    image_data = image_data / 255.

    #image_name = image_path.split('/')[-1]
    #image_name = image_name.split('.')[0]

    images_data = []
    for i in range(1):
        images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

    
    infer = saved_model_loaded.signatures['serving_default']
    batch_data = tf.constant(images_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=0.45,
        score_threshold=0.25

    )

       
    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
    original_h, original_w, _ = original_image.shape
        
    bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
        
    # hold all detection data in one variable
    pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
       
    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)

    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())
        
    # custom allowed classes (uncomment line below to allow detections for only people)
    #allowed_classes = ['person']

   
    crop_path = os.path.join(os.getcwd(), 'detections', 'crop')
    try:
        os.mkdir(crop_path)
    except FileExistsError:
        pass
    
    crop_objects(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), pred_bbox, crop_path, allowed_classes,1)

      

    image = utils.draw_bbox(original_image, pred_bbox, allowed_classes = allowed_classes)

    image = Image.fromarray(image.astype(np.uint8))
        
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    cv2.imwrite('./detections/' + 'detection' + '.png', image)


    
    with open(fp,'rb') as f:
      test_result = recognize_plate(f);
      temp_result = test_result['results']
      plate = temp_result[0]['plate']
      print(plate)


    return test_result

    



