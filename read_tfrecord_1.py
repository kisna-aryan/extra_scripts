import os
import re
import zipfile

import numpy as np
import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

(train_dataset, val_dataset), dataset_info = tfds.load("coco/2017", split=["train", "validation"], with_info=True, data_dir="K:\DL_dataset\data")
print(dataset_info)
# ds_train = tfds.load('train.tfrecord', data_dir='K:\DL_git\flirRGBtfrecord\flirRGBtfrecord')

# ds_train = tf.data.TFRecordDataset(tf.io.gfile.glob('K:\DL_git\flirRGBtfrecord\flirRGBtfrecord/train.tfrecord'))
print(train_dataset.take(10))