
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True

import numpy as np
from pathlib import Path
import tensorflow as tf 

from variables import*

np.random.seed(seed)

def get_class_labels():
    return os.listdir(train_dir)

def move_image(source_path, destination_path):
    Path(source_path).rename(destination_path)

def move_from_train(data_dir, img_arr, label, source_label_dir):
    destination_label_dir = os.path.join(data_dir, label)

    if not os.path.exists(destination_label_dir):
        os.makedirs(destination_label_dir)

    for img in img_arr:
        destination_path = os.path.join(destination_label_dir, img)
        source_path = os.path.join(source_label_dir, img)
        move_image(source_path, destination_path)

def split_train_val_test():
    if not os.path.exists(val_dir) or not os.path.exists(test_dir):

        os.makedirs(val_dir)
        os.makedirs(test_dir)

        labels = get_class_labels()

        for label in labels:
            source_label_dir = os.path.join(train_dir, label)
            image_arr = os.listdir(source_label_dir)

            Ntest = int(test_size * len(image_arr))
            Nval  = int(val_size * len(image_arr))

            np.random.shuffle(image_arr)
            val_img_arr = image_arr[-(Ntest+Nval):-Ntest]
            test_img_arr = image_arr[-Ntest:]
        
            move_from_train(val_dir, val_img_arr, label, source_label_dir)
            move_from_train(test_dir, test_img_arr, label, source_label_dir)

def get_filenames(data_dir):
    labels = get_class_labels()
    filenames = []
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        absolute_label_dir = [os.path.join(label_dir, img_path) for img_path in os.listdir(label_dir)]
        filenames.extend(absolute_label_dir)
    return filenames

def get_label_from_filename(filename):
    return os.path.split(filename)[-2]

def decode_audio(filename):
    audio_binary = tf.io.read_file(filename)
    audio, sample_rate = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)
