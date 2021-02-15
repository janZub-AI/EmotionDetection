# ZCA whitening
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os
import pandas as pd
import tensorflow as tf
from utils import Utils


def get_data_for_category(dir):
	a=[]
	_, label = os.path.split(dir)
	for k,j in enumerate(os.listdir(dir)):
		a.append((f'{dir}/{j}',label))

	df = pd.DataFrame(a,columns=['filename','class']).sample(frac=1).reset_index(drop=True)
	return df

def generate_aug_data(df,emotion, datagen, aug_emotion_path, guid = 1):
	created = 0
	limit = len(df.get('filename'))
	if(emotion == 'happy'): limit = limit * 2
	if(emotion in ['neutral', 'sad', 'angry', 'fear', 'surprise']): limit = limit * 3
	if(emotion == 'disgust'): limit = limit * 3 * 4
	
	for batch in datagen.flow_from_dataframe(df, target_size= (48,48), color_mode = 'grayscale', batch_size=64, save_to_dir=aug_emotion_path, save_prefix=f'aug_{guid}', save_format='jpg'):
		created = created + 64
		if(created > limit):
			return

def view_batch(batch):
	batch_size = len(batch[0])
	for i in range(0, batch_size):
		pyplot.subplot(batch_size/10, 10 + (batch_size % 10), i + 1)
		pyplot.imshow(batch[0][i].reshape(48, 48), cmap=pyplot.get_cmap('gray'))
	pyplot.show()

def process_folder(folder):
	aug_path = os.path.join(dirname, f'aug2_{folder}')
	if not os.path.exists(aug_path):
		os.makedirs(aug_path)

	folder_dir = os.path.join(dirname, folder)	

	for emotion in os.listdir(folder_dir):
		aug_emotion_path = os.path.join(aug_path, emotion)
		if not os.path.exists(aug_emotion_path):
			os.makedirs(aug_emotion_path)
	
		emotion_dir = os.path.join(folder_dir, emotion)
		data = get_data_for_category(emotion_dir)

		generate_aug_data(data, emotion, datagen, aug_emotion_path)


dirname = os.path.dirname(__file__)

# in case of zca_whitening 
'''files = Utils.limit_data(os.path.join(dirname, 'train'))

imgs = []

for f in files.get('file'):
	image = tf.keras.preprocessing.image.load_img(
            f, color_mode='grayscale', target_size=(48,48),
            interpolation='nearest'
        )
	img_array = tf.keras.preprocessing.image.img_to_array(image)
	img_array = tf.cast(img_array ,tf.float32)
	imgs.append(img_array)
'''
shift = 0.1
datagen = ImageDataGenerator(rotation_range=45, vertical_flip=True, horizontal_flip=True, width_shift_range=shift, height_shift_range=shift)
#datagen.fit(imgs)



folders = ['train', 'dev', 'test']

process_folder('train')