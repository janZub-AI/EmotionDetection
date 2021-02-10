# baseline cnn model for mnist
from keras.callbacks import TensorBoard
from kerastuner.tuners import RandomSearch

import datetime

from utils import Utils
from small_images import SmallImagesModel, SmallImagesHP
from rename_tensorboard import FileManager

def run_tuner(hypermodel, hp):
    # load dataset
    trainX, trainY, testX, testY = Utils.load_handwriting_dataset()
    # prepare pixel data
    trainX, testX = Utils.prep_pixels(trainX, testX)

    tuner = RandomSearch(
        hypermodel,
        objective = 'val_accuracy',
        max_trials = TUNER_SETTINGS['max_trials'],      
        metrics=['accuracy'], 
        loss='categorical_crossentropy',
        hyperparameters = hp,
        executions_per_trial = TUNER_SETTINGS['executions_per_trial'],
        directory = TUNER_SETTINGS['log_dir'],     
        project_name = 'mist_tuner')

    tuner.search(trainX, trainY,
                batch_size = TUNER_SETTINGS['batch_size'],
                callbacks = [TUNER_SETTINGS['tb_callback']],
                epochs = TUNER_SETTINGS['epochs'],
                validation_data = (testX, testY))
   
    tuner.search_space_summary()
    tuner.results_summary()
    models = tuner.get_best_models(num_models=5)

    print(models)

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tb_callback = TensorBoard(
        log_dir = log_dir,
        histogram_freq = 1,
        embeddings_freq = 1,
        write_graph = True,
        update_freq = 'batch')

TUNER_SETTINGS = {
    'log_dir' : log_dir,
    'batch_size' : 32,  
    'epochs' : 5,
    'max_trials' : 3,
    'executions_per_trial' : 1,
    'tb_callback' : tb_callback
    }

hyperparameters = SmallImagesHP(init_min=128, init_max=256)

hp = SmallImagesModel.define_hp(hyperparameters)
hypermodel = SmallImagesModel(num_classes = 10, input_shape = (28, 28, 1))

run_tuner(hypermodel, hp)
input("Press Enter to continue...")
FileManager.rename_files(TUNER_SETTINGS['log_dir'], hypermodel.generate_model_name)