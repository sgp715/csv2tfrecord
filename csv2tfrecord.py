# import tensorflow as tf
import argparse

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(images_directiry, labels_file, out_file):
    print(images_directiry)
    print(labels_file)
    print(out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images_directory")
    parser.add_argument("labels_file")
    parser.add_argument("--out")
    args = parser.parse_args()
    if args.out == None:
        main(args.images_directory, args.labels_file, "out.tfrecords")
    else:
        main(args.images_directory, args.labels_file, args.out)