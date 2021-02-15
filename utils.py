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
import matplotlib.pyplot as plt

dirname = os.path.dirname(__file__)
class Utils():
    # load train and test dataset
    def load_emotion_dataset(subfolder, take = 1024, batch_size = 32, aug_data = False):
        df = Utils.limit_data(subfolder, take = take, aug_data = aug_data)

        onehot_encoded = Utils.get_values_hot(df.get('label'))     
        labels = tf.convert_to_tensor(onehot_encoded)
        image_paths = tf.convert_to_tensor(df.get('file'), dtype=tf.string)

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        dataset = dataset.map(Utils.parse_function, num_parallel_calls = 10)
        dataset = dataset.batch(batch_size)
        dataset.shuffle(100000)
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


    def limit_data(subfolder, aug_data, skip=0, take=1024):
       
        def get_pd(dir):
            a=[]
            for i in os.listdir(dir):
                for k,j in enumerate(os.listdir(dir+'/'+i)):
                    if k>=take:continue
                    if k<skip:continue

                    a.append((f'{dir}/{i}/{j}',i))
            return a
   
        dir = os.path.join(dirname, subfolder)
        data = get_pd(dir)

        if(aug_data):
            dir = os.path.join(dirname, f'aug2_{subfolder}')
            data = data + get_pd(dir)

        df = pd.DataFrame(data,columns=['file','label']).sample(frac=1).reset_index(drop=True)
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
    
    def plot_images_data():
        row, col = 48, 48
        classes = 7

        def count_exp(path, set_):
            dict_ = {}
            for expression in os.listdir(path):
                dir_ = os.path.join(path, expression)
                dict_[expression] = len(os.listdir(dir_))
            df = pd.DataFrame(dict_, index=[set_])
            return df

        train_count = count_exp(os.path.join(dirname, 'train'), 'train')        
        aug_train_count = count_exp(os.path.join(dirname, 'aug2_train'), 'aug_train')
        dev_count = count_exp(os.path.join(dirname, 'dev'), 'dev')  
        aug_dev_count = count_exp(os.path.join(dirname, 'aug2_dev'), 'aug_dev')
        #test_count = count_exp(os.path.join(dirname, 'test'), 'test')  
        #aug_test_count = count_exp(os.path.join(dirname, 'aug_test'), 'aug_test')
        print(train_count)
        print(aug_train_count)
        print(dev_count)
        print(aug_dev_count)
        #print(test_count)
        #print(aug_test_count)

        #train_count.transpose().plot(kind='bar')
        #aug_train_count.transpose().plot(kind='bar')
        #dev_count.transpose().plot(kind='bar')
        #aug_dev_count.transpose().plot(kind='bar')
        #test_count.transpose().plot(kind='bar')
        #aug_test_count.transpose().plot(kind='bar')

        plt.show()

#Utils.plot_images_data()