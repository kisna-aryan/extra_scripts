import tensorflow as tf

raw_dataset = tf.data.TFRecordDataset("/media/kisna/nano_ti_data/DL_dataset/data/coco/2017/1.1.0/coco-train.tfrecord-00000-of-00256")
# raw_dataset = tf.data.TFRecordDataset("/media/kisna/nano_ti_data/DL_dataset/data/final.data-00001-of-00002")



for raw_record in raw_dataset.take(1):
    
    for key in raw_record.keys():
        if key != "image":
            print(f"{key}: {raw_record[key]}")
    
    # example = tf.train.Example()
    # example.ParseFromString(raw_record.numpy())
    # print(example)