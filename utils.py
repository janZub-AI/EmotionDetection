import tensorflow as tf
import os
import pandas as pd
import numpy as np
import datetime
import time
from keras.utils import to_categorical
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

dirname = os.path.dirname(__file__)
class Utils():
    # load train and test dataset
    def load_emotion_dataset(subfolder, take = 1024, batch_size = 32):
        df = Utils.limit_data(os.path.join(dirname, subfolder), take = take)

        onehot_encoded = Utils.get_values_hot(df.get('label'))     
        labels = tf.convert_to_tensor(onehot_encoded)
        image_paths = tf.convert_to_tensor(df.get('file'), dtype=tf.string)

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(Utils.parse_function, num_parallel_calls = 10)
        dataset = dataset.batch(batch_size)
        dataset.shuffle(10000)
        dataset = dataset.cache() 
        
        return dataset

    def parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=1)
        image = tf.cast(image_decoded/255, tf.float32)
        image = tf.image.resize(image, [48, 48])
        return image, label

    def get_values_hot(values):
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(values)
        onehot_encoder = OneHotEncoder(sparse=False, dtype=np.int32)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        return onehot_encoder.fit_transform(integer_encoded)



    def map_fn(filename, label):
        print(filename)
        # path/label represent values for a single example
        image = tf.keras.preprocessing.image.load_img(
            filename, color_mode='grayscale', target_size=(48,48),
            interpolation='nearest'
        )
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        # color normalization - just an example    
        img_array = tf.cast(img_array/255. ,tf.float32)
        return img_array, label


    def prep_pixels(image,label):
        image = tf.cast(image/255. ,tf.float32)
        return image,label


    def limit_data(dir,skip=0,take=1024):
        a=[]
        for i in os.listdir(dir):
            for k,j in enumerate(os.listdir(dir+'/'+i)):
                if k>=take:continue
                if k<skip:continue

                a.append((f'{dir}/{i}/{j}',i))

        df = pd.DataFrame(a,columns=['file','label']).sample(frac=1).reset_index(drop=True)
        return df

    def move_files(source, dest):
        skip = 100
        for i in os.listdir(os.path.join(dirname, source)):
            i_path = os.path.join(dirname, dest, i) 

            files_count = len([name for name in os.listdir(i_path) if os.path.isfile(os.path.join(i_path, name))])
            if not os.path.exists(i_path):
                os.makedirs(i_path)
            for k,j in enumerate(os.listdir(i_path)):
                if k>=files_count * 0.10 + skip:break
                if k<skip:continue

                print(f'{dir}/train/{i}/{j}')
                os.replace(os.path.join(i_path, j), os.path.join(dirname, dest, i, j))

    def load_data_using_tfdata(dir, take = 1024):
        """
        Load the images in batches using Tensorflow (tfdata).
        Cache can be used to speed up the process.
        Faster method in comparison to image loading using Keras.
        Returns:
        Data Generator to be used while training the model.
        """
        def parse_image(file_path):
            parts = tf.strings.split(file_path, os.path.sep)
            class_names = np.array(os.listdir(dirname + '/test'))
            # The second to last is the class-directory
            label = parts[-2] == class_names

            # load the raw data from the file as a string
            img = tf.io.read_file(file_path)
            # convert the compressed string to a 3D uint8 tensor
            img = tf.image.decode_jpeg(img, channels=1)
            # Use `convert_image_dtype` to convert to floats in the [0,1] range
            img = tf.image.convert_image_dtype(img/255, tf.float32)
            # resize the image to the desired size.
            img = tf.image.resize(img, [80, 80])
            return img, tf.argmax(label)

        def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
            # If a small dataset, only load it once, and keep it in memory.
            # use `.cache(filename)` to cache preprocessing work for datasets
            # that don't fit in memory.
            if cache:
                if isinstance(cache, str):
                    ds = ds.cache(cache)
                else:
                    ds = ds.cache()
            ds = ds.shuffle(buffer_size=shuffle_buffer_size)
            # Repeat forever
            ds = ds.repeat()
            ds = ds.batch(32)
            # `prefetch` lets the dataset fetch batches in the background
            # while the model is training.
            ds = ds.prefetch(buffer_size=AUTOTUNE)
            return ds

        data_generator = {}
        dataset_location = os.path.join(dirname, dir)
        folders = os.listdir(dataset_location)

        for folder in folders:
            dir_extend = os.path.join(dataset_location, folder, '*')
            print(dir_extend)
            list_ds = tf.data.Dataset.list_files(dir_extend).take(take)
            print(list_ds)
            AUTOTUNE = tf.data.experimental.AUTOTUNE
            # Set `num_parallel_calls` so that multiple images are
            # processed in parallel
            labeled_ds = list_ds.map(
                parse_image, num_parallel_calls=AUTOTUNE)
            # cache = True, False, './file_name'
            # If the dataset doesn't fit in memory use a cache file,
            # eg. cache='./data.tfcache'
            data_generator[folder] = prepare_for_training(
                labeled_ds, cache='./data.tfcache', shuffle_buffer_size = 10000)
        return data_generator
