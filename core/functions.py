import os
import cv2
from core.utils import read_class_names
from core.config import cfg
# import easyocr 

def crop_objects(img, data, path, allowed_classes,cnt):
    boxes, scores, classes, num_objects = data
    
    class_names = read_class_names(cfg.YOLO.CLASSES)
    #create dictionary to hold count of objects for image name
    counts = dict()
    for i in range(num_objects):
        # get count of class for part of image name
        class_index = int(classes[i])
        class_name = class_names[class_index]
        if class_name in allowed_classes:
            counts[class_name] = counts.get(class_name, 0) + 1
            
            # get box coords
            xmin, ymin, xmax, ymax = boxes[i]
            # crop detection from image (take an additional 5 pixels around all edges)
            cropped_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
            # construct image name and join it to path for saving crop properly
            img_name = class_name + '_' + str(cnt)+str(i) + '.png'
            # img_name = class_name + '_' + str(counts[class_name]) + '.png'
            img_path = os.path.join(path, img_name )
            # save image
            cv2.imwrite(img_path, cropped_img)
        else:
            continue

    print(counts)
    print(num_objects)

# def preprocess_detect(image_path,cnt):
  
#   total_read = []
#   image = cv2.imread(image_path)
#   # grayscale region within bounding box
#   gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#   # resize image to double the original size as tesseract does better with certain text size
#   blur = cv2.resize(gray, None, fx = 2, fy = 2, interpolation = cv2.INTER_CUBIC)

#   '''
#   image = cv2.imread(image_path)
#   gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#   #nr_img = cv2.bilateralFilter(gray_img, 9, 75, 75)
#   #eh_img = cv2.equalizeHist(gray_img)
#   '''

#   img_name = 'testing'+str(cnt)+'.png'
#   cv2.imwrite('./detections/crop_detected_testing'+img_name, blur)

#   reader = easyocr.Reader(['en'])
#   #im = PIL.Image.open('./detections/testing.png')
#   bounds = reader.readtext(blur,detail = 0)
#   total_read.append(bounds)
#   print(total_read)


