
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').disabled = True

import numpy as np
from pathlib import Path
import tensorflow as tf 

from variables import*

np.random.seed(seed)
autotune = tf.data.experimental.AUTOTUNE

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
    # return tf.strings.split(filename, os.path.sep)[-2]
    return os.path.split(filename)[-2]

def decode_audio(filename):
    audio_binary = tf.io.read_file(filename)
    audio, sample_rate = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_waveform_and_label(filename):
    label = get_label_from_filename(filename)
    audio_data = decode_audio(filename)
    return audio_data, label

def get_spectrum(audio):
    padding = tf.zeros(
                    [audio_tensor_length] - tf.shape(audio), 
                    dtype=tf.float32
                    )

    audio = tf.cast(audio, tf.float32)
    padded = tf.concat([audio, padding], 0)
    spectrum = tf.signal.stft(
                            padded, 
                            frame_length=frame_length, 
                            frame_step=frame_step
                            )
    return tf.abs(spectrum)

def encode_labels(label):
    all_labels = np.array(get_class_labels())
    return tf.argmax(label == all_labels)

def get_inputs_and_outputs(filename):
    audio, label = get_waveform_and_label(filename)
    spectrum = get_spectrum(audio)
    spectrum = tf.expand_dims(spectrum, -1) # Add channel Dimension
    # label = encode_labels(label)
    return spectrum, tf.convert_to_tensor(label, dtype=tf.string)

def create_data_split(data_dir, autotune=autotune):
    filenames = get_filenames(data_dir)
    data_files  = tf.data.Dataset.from_tensor_slices(filenames)
    data_split = data_files.map(get_inputs_and_outputs,  num_parallel_calls=autotune)
    return data_split

def load_data():
    train_data = create_data_split(train_dir)
    val_data = create_data_split(val_dir)
    test_data = create_data_split(test_dir)
    print(list(test_data)[0])

load_data()