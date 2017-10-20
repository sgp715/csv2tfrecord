import tensorflow as tf
import argparse
from os import walk
import imghdr
import numpy as np
from PIL import Image

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
   return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_tf_example(example):

  filename = str.encode(example["filename"])
  encoded_image_data = example["bytes"]
  image_format = str.encode(example["format"])
  height = example["height"]
  width = example["width"]

  xmins = [ xmin / width for xmin in example["xmins"] ]
  xmaxs = [ xmax / width for xmax in example["xmaxs"] ]
  ymins = [ ymin / width for ymin in example["ymins"] ]
  ymaxs = [ ymax / width for ymax in example["ymaxs"] ]
  classes_text = example["classes_text"]
  # classes = example["classes"]

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
      'image/filename': bytes_feature(filename),
      'image/source_id': bytes_feature(filename),
      'image/encoded': bytes_feature(encoded_image_data),
      'image/format': bytes_feature(image_format),
      'image/object/bbox/xmin': float_list_feature(xmins),
      'image/object/bbox/xmax': float_list_feature(xmaxs),
      'image/object/bbox/ymin': float_list_feature(ymins),
      'image/object/bbox/ymax': float_list_feature(ymaxs),
      'image/object/class/text': bytes_list_feature(classes_text),
      # 'image/object/class/label': int64_list_feature(classes),
  }))
  return tf_example

def get_examples(images_directory, labels_file):

    files = None
    for (dirpath, dirnames, filenames) in walk(images_directory):
        files = filenames

    examples = []
    for f in files:
        example = {}

        example["filename"] = f
        img = np.array(Image.open(images_directory + f))
        example["bytes"] = img.tostring()
        example["height"] = img.shape[0]
        example["width"] = img.shape[1]
        example["format"] = imghdr.what(images_directory + f)

        example["xmins"] = []
        example["xmaxs"] = []
        example["ymins"] = []
        example["ymaxs"] = []
        example["classes_text"] = []
        # example["classes"] = []
        labels = open(labels_file, 'r')
        labels.readline() # throw away
        line = labels.readline()
        while line:
            vals = line.split(',')
            if vals[1] == f:
                example["xmins"].append(vals[2])
                example["xmaxs"].append(vals[3])
                example["ymins"].append(vals[4])
                example["ymaxs"].append(vals[5])
            line = labels.readline()
        labels.close()
        examples.append(example)
    return examples

def main(images_directory, labels_file, out_file):

    writer = tf.python_io.TFRecordWriter(out_file)

    examples = get_examples(images_directory, labels_file)

    for example in examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images_directory")
    # parser.add_argument("images_width")
    # parser.add_argument("images_height")
    parser.add_argument("labels_file")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.images_directory[-1] != '/':
        args.images_directory += '/'
    if args.out == None:
        args.out = "out.tfrecords"
    main(args.images_directory, args.labels_file, args.out)