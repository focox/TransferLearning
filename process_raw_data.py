import glob
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tqdm

INPUT_DATA = '../flower_photos'
OUTPUT_TRAIN = './train_data.npy'
OUTPUT_VALIDATION = './validation_data.npy'
OUTPUT_TEST = './test_data.npy'

VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


def create_image_lists(sess, testing_percentage, validation_percentage):
    sub_dirs_files = [x for x in os.walk(INPUT_DATA)]

    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    for sub in tqdm.tqdm(sub_dirs_files[1:]):
        # sub_dirs[0] is the root dir
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        current_path = sub[0]
        sub_files = sub[2]
        for file in tqdm.tqdm(sub_files):
            if file.split('.')[-1] in extensions:
                image_raw_data = gfile.FastGFile(os.path.join(current_path, file), 'rb').read()
                image = tf.image.decode_jpeg(image_raw_data)
                if image.dtype != tf.float32:
                    image = tf.image.convert_image_dtype(image, tf.float32)
                    image = tf.image.resize_images(image, [299, 299])
                    image_value = sess.run(image)

                    chance = np.random.randint(100)
                    if chance < validation_percentage:
                        validation_images.append(image_value)
                        validation_labels.append(current_label)
                    elif chance < (testing_percentage + validation_percentage):
                        testing_images.append(image_value)
                        testing_labels.append(current_label)
                    else:
                        training_images.append(image_value)
                        training_labels.append(current_label)
        current_label += 1

    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    processed_train = np.asarray([training_images, training_labels])
    np.save(OUTPUT_TRAIN, processed_train)
    del processed_train
    processed_validation = np.asarray([validation_images, validation_labels])
    np.save(OUTPUT_VALIDATION, processed_validation)
    del processed_validation
    processed_test = np.asarray([testing_images, testing_labels])
    np.save(OUTPUT_TEST, processed_test)


def main():
    with tf.Session() as sess:
        create_image_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)


if __name__ == '__main__':
    main()
