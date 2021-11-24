import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
import glob
import csv

flags.DEFINE_string('classes', './data/labels/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './weights/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_list('images', None, 'list with paths to input images')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_string('testing_folder', None, 'folder path to input images')
flags.DEFINE_boolean('test_r', False, 'test for testing sub folders')
flags.DEFINE_string('output', './detections/', 'path to output folder')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_boolean('save_detection', True, 'detection is saved or not')


def getFileNameFromPath(path='', isForword=True):
    splitted = path.split('/') if isForword else path.split('\\')
    return splitted[len(splitted) - 1]


def main(_argv):
    if FLAGS.testing_folder is None and FLAGS.images is None:
        print('No test data provided')
        return

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    print('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    print('classes loaded')

    if FLAGS.testing_folder:
        print('Testing for a folder.........')
        testing_folder_path = FLAGS.testing_folder

        images = []
        if FLAGS.test_r:
            print('Testing recursively.........')
            testing_sub_folders = glob.glob(testing_folder_path)
            for testing_sub_folder in testing_sub_folders:
                sub_folder_images = glob.glob(testing_sub_folder + '*.jpg')
                images.extend(sub_folder_images)
        else:
            images = glob.glob(testing_folder_path)

        raw_images = []
        print('Decoding images...')
        for image in images:
            decode_t1 = time.time()
            img_raw = tf.image.decode_image(
                open(image, 'rb').read(), channels=3)
            decode_t2 = time.time()
            print('Decode time: {}'.format(decode_t2 - decode_t1))
            raw_images.append(img_raw)
    elif FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, FLAGS.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        raw_images = []
        images = FLAGS.images
        for image in images:
            img_raw = tf.image.decode_image(
                open(image, 'rb').read(), channels=3)
            raw_images.append(img_raw)
    num = 0
    with open('./detections/results.csv', 'w', encoding='UTF8', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            ['Image Name', 'Detection saved to', 'Exec time', 'Class 1', 'Confidence', 'Class 2', 'Confidence',
             'Class 3', 'Confidence'])

        for raw_img in raw_images:
            num += 1
            img = tf.expand_dims(raw_img, 0)
            img = transform_images(img, FLAGS.size)

            t1 = time.time()
            boxes, scores, classes, nums = yolo(img)
            t2 = time.time()
            logging.info('time: {}'.format(t2 - t1))

            image_name = getFileNameFromPath(images[num - 1])
            print('Image: {} detections:'.format(image_name))
            for i in range(nums[0]):
                print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))

            if FLAGS.save_detection:
                img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
                img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
                cv2.imwrite(FLAGS.output + 'detection' + str(num) + '.jpg', img)
                print('output saved to: {}'.format(FLAGS.output + 'detection' + str(num) + '.jpg'))

            csv_row = [image_name, 'detection' + str(num) + '.jpg', (t2 - t1)]
            for i in range(nums[0]):
                csv_row.append(class_names[int(classes[0][i])])
                csv_row.append(np.array(scores[0][i]))

            csv_writer.writerow(csv_row)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
