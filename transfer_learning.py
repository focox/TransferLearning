import tensorflow as tf
import glob
import os
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.python.platform import gfile

import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

TRAIN_DATA = './train_data.npy'
VALIDATE_DATA = './validation_data.npy'
TEST_DATA = './testing_data.npy'

TRAIN_FILE = './model/'
CKPT_FILE = './inception_v3.ckpt'

LEARNING_RATE = 0.0001
STEPS = 300000
BATCH = 32
N_CLASSES = 5

CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'
TRAINABLE_SCOPES = 'InceptionV3/Logits, InceptionV3/AuxLogits'


def get_tuned_variables():
    exclusions = [scope.strip() for scope in CHECKPOINT_EXCLUDE_SCOPES.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    return variables_to_restore


def get_trainable_variables():
    scopes = [scope.strip() for scope in TRAINABLE_SCOPES.split(',')]
    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
    return variables_to_train


def main(argv=None):
    training_data = np.load(TRAIN_DATA)
    training_images = training_data[0]
    n_training_example = len(training_images)
    training_labels = training_data[1]

    validation_data = np.load(VALIDATE_DATA)
    validation_images = validation_data[0]
    validation_labels = validation_data[1]

    testing_data = np.load(TEST_DATA)
    testing_images = testing_data[0]
    testing_labels = testing_data[1]

    print('%d training examples, %d validation examples and %d testing examples.' % (
        n_training_example, len(validation_labels), len(testing_labels)))

    images = tf.placeholder(tf.float32, [None, 299, 299, 3], name='input_images')
    labels = tf.placeholder(tf.int64, [None], name='labels')

    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, _ = inception_v3.inception_v3(images, num_classes=N_CLASSES)
        trainable_variables = get_trainable_variables()
        tf.losses.softmax_cross_entropy(tf.one_hot(labels, N_CLASSES), logits, weights=1.0)
        train_step = tf.train.RMSPropOptimizer(LEARNING_RATE).minimize(tf.losses.get_total_loss())

    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(logits, 1), labels)
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    load_fn = slim.assign_from_checkpoint_fn(CKPT_FILE, get_tuned_variables(), ignore_missing_vars=True)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print('Loading tuned variables from %s' % CKPT_FILE)
        load_fn(sess)

        start = 0
        end = BATCH
        for i in range(STEPS):
        #     print('start-end', start, end)
            sess.run(train_step, feed_dict={images: np.array(training_images[start:end].tolist()),
                                            labels: training_labels[start:end]})
            if i % 30 == 0 or i + 1 == STEPS:
                saver.save(sess, TRAIN_FILE, global_step=i)
                validation_accuracy = sess.run(evaluation_step,
                                               feed_dict={images: np.array(validation_images[:BATCH].tolist()),
                                                          labels: validation_labels[:BATCH]})
                print('Step %d: Validation accuracy = %.1f%%' % (i, validation_accuracy * 100.0))

            start = end
            if start == n_training_example:
                start = 0
            end = start + BATCH
            if end > n_training_example:
                end = n_training_example
        test_accuracy = sess.run(evaluation_step, feed_dict={images: np.array(testing_images.tolist()),
                                                             labels: testing_labels})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()
