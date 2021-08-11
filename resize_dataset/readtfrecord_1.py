import os
import json
import pprint
import tensorflow as tf
import matplotlib.pyplot as plt
# from fuction_fn_odapi import *
from fuctions_fn import *

datasettype = 'test'

root_dir = "/media/kisna/nano_ti_data/DL_git/FLIRrgb/flirRGB"
tfrecords_dir = f"/media/kisna/nano_ti_data/DL_git/FLIRrgb_keras/{datasettype}"
images_dir = os.path.join(root_dir, f"{datasettype}/thermal_8_bit")
annotations_dir = os.path.join(root_dir, f"{datasettype}")
annotation_file = os.path.join(annotations_dir, 'thermal_annotations.json')


with open(annotation_file, "r") as f:
    annotations = json.load(f)["annotations"]

print(f"Number of images: {len(annotations)}")

pprint.pprint(annotations[60])

num_samples = 4096
num_tfrecods = len(annotations) // num_samples
if len(annotations) % num_samples:
    num_tfrecods += 1  # add one record if there are any remaining samples

print(num_tfrecods)

if not os.path.exists(tfrecords_dir):
    os.makedirs(tfrecords_dir)  # creating TFRecords output folder
    print('tfrecords_dir created')
    print(tfrecords_dir)
print('tfrecords_dir existed')

for tfrec_num in range(num_tfrecods):
    samples = annotations[(tfrec_num * num_samples) : ((tfrec_num + 1) * num_samples)]

    with tf.io.TFRecordWriter(
        tfrecords_dir + "/file_%.2i-%i.tfrec" % (tfrec_num, len(samples))
    ) as writer:
        for sample in samples:
            filename_temp = sample['image_id'] + 1
            filename = 'FLIR_video_' + f"{filename_temp:05d}"
            image_path = f"{images_dir}/{filename}.jpeg"
            image = tf.io.decode_jpeg(tf.io.read_file(image_path))
            example = create_example(image, image_path, sample)
            writer.write(example.SerializeToString())



raw_dataset = tf.data.TFRecordDataset(f"{tfrecords_dir}/file_00-{num_samples}.tfrec")
# raw_dataset = tf.data.TFRecordDataset(f"/media/kisna/nano_ti_data/DL_git/flirRGBtfrecord/flirRGBtfrecord/train.tfrecord")
parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

for features in parsed_dataset.take(1):
    for key in features.keys():
        if key != "image":
            print(f"{key}: {features[key]}")

    print(f"Image shape: {features['image'].shape}")
    plt.figure(figsize=(7, 7))
    plt.imshow(features["image"].numpy())
    plt.show()