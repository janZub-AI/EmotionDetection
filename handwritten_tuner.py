# baseline cnn model for mnist
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler
from kerastuner.tuners import RandomSearch

import datetime

from utils import Utils
from small_images import SmallImagesModel, SmallImagesHP
from rename_tensorboard import FileManager
from callbacks.early_stopping import EarlyStoppingAtMinLoss


def run_tuner(hypermodel, hp):
    # load dataset
    train_data_per_category = TUNER_SETTINGS['batches_per_category'] * TUNER_SETTINGS['batch_size']
    validation_data_per_category = TUNER_SETTINGS['batches_for_validation'] * TUNER_SETTINGS['batch_size']

    train_dataset = Utils.load_emotion_dataset('train', train_data_per_category, TUNER_SETTINGS['batch_size'])
    test_dataset = Utils.load_emotion_dataset('dev', validation_data_per_category, TUNER_SETTINGS['batch_size'])
    
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
    
    tb_callback = TensorBoard(
            log_dir = log_dir,
            histogram_freq = 1,
            embeddings_freq = 1,
            update_freq = 'epoch')

    tuner.search(train_dataset, validation_data = test_dataset,
                batch_size = TUNER_SETTINGS['batch_size'],
                callbacks = TUNER_SETTINGS['callbacks'] + [tb_callback],
                epochs = TUNER_SETTINGS['epochs']
                )
   
    tuner.search_space_summary()
    tuner.results_summary()
    models = tuner.get_best_models(num_models=5)

    print(models)

log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

def scheduler(epoch, lr):
  if epoch % 5 == 0:
    lr = lr*0.8
  return lr


lr_callback = LearningRateScheduler(scheduler)
es_callback = EarlyStoppingAtMinLoss(patience = 3, epoch = 5)

TUNER_SETTINGS = {
    'log_dir' : log_dir,    
    'batch_size' : 32,  
    'batches_per_category' : 100000,
    'batches_for_validation' : 100000,
    'epochs' : 50,
    'max_trials' : 5,
    'executions_per_trial' : 1,
    'callbacks' : [es_callback, lr_callback]
    }

hyperparameters = SmallImagesHP(
    init_min = 96, 
    init_max = 96, 
    cnn_layers_min = 2,
    cnn_layers_max = 2,
    cnn_min = 224,
    cnn_max = 224,
    cnn2_layers_min = 2,
    cnn2_layers_max = 2,
    cnn2_min = 160,
    cnn2_max = 160,
    dense_min = 600,
    dense_max = 1000,
    dense_step = 50,
    dense2_min = 250,
    dense2_max = 700,
    dense2_step = 50,
    dropout = [0.5],
    learning_rate = [0.001]
)

hp = SmallImagesModel.define_hp(hyperparameters)
hypermodel = SmallImagesModel(num_classes = 7, input_shape = (48, 48, 1))

run_tuner(hypermodel, hp)
input("Press Enter to continue...")
FileManager.rename_files(TUNER_SETTINGS['log_dir'], hypermodel.generate_model_name)