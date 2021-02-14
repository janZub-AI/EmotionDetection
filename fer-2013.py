from keras.callbacks import TensorBoard, ModelCheckpoint, LearningRateScheduler
from kerastuner.tuners import RandomSearch

import datetime

from utils import Utils
from small_images import SmallImagesModel, SmallImagesHP
from rename_tensorboard import FileManager
from callbacks.early_stopping import EarlyStoppingAt

# global for logging and unique name
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# callbacks
def get_model_checkout(current_time):
    return ModelCheckpoint(
            filepath = f'models_checkpoint/{current_time}/'+'{epoch:02d}-{val_loss:.5f}.hdf5',
            save_weights_only=False,
            verbose = 1,
            monitor='val_loss',
            mode='auto',
            period = 5,
            save_best_only=True)
def get_tensorboard(log_dir):
    return TensorBoard(
            log_dir = log_dir,
            histogram_freq = 1,
            embeddings_freq = 1,
            update_freq = 'epoch')
def get_lr_scheduler():
    def scheduler(epoch, lr):
        if epoch % 5 == 0:
            lr = lr*0.8
        return lr

    return LearningRateScheduler(scheduler)
def get_early_stopping():
    return EarlyStoppingAt(patience = 3, ignored_epoch = 5, stop_at = 'val_loss')

# helpers
def load_data():
    train_data_per_category = TUNER_SETTINGS['batches_per_category'] * TUNER_SETTINGS['batch_size']
    validation_data_per_category = TUNER_SETTINGS['batches_for_validation'] * TUNER_SETTINGS['batch_size']

    train_dataset = Utils.load_emotion_dataset('train', train_data_per_category, TUNER_SETTINGS['batch_size'])
    test_dataset = Utils.load_emotion_dataset('dev', validation_data_per_category, TUNER_SETTINGS['batch_size'])

    return train_dataset, test_dataset

# runner
def run_tuner(hypermodel, hp):   
    # load dataset
    train_dataset, test_dataset = load_data()

    # init tensorboard here so each run will have folder 
    # which we can rename based on trial_id
    tb_callback = get_tensorboard(TUNER_SETTINGS['log_dir'])

    tuner = RandomSearch(
        hypermodel,
        objective = TUNER_SETTINGS['objective'],
        max_trials = TUNER_SETTINGS['max_trials'],      
        metrics=['accuracy'], 
        loss='categorical_crossentropy',
        hyperparameters = hp,
        executions_per_trial = TUNER_SETTINGS['executions_per_trial'],
        directory = TUNER_SETTINGS['log_dir'],     
        project_name = 'fer-2013')


    
    tuner.search(train_dataset, validation_data = test_dataset,
                batch_size = TUNER_SETTINGS['batch_size'],
                callbacks = TUNER_SETTINGS['callbacks'] + [tb_callback],
                epochs = TUNER_SETTINGS['epochs']
                )


mc_callback = get_model_checkout(current_time)
lr_callback = get_lr_scheduler()
es_callback = get_early_stopping()

# params
TUNER_SETTINGS = {
    'log_dir' : f'logs/{current_time}',    
    'batch_size' : 32,  
    'batches_per_category' : 10000,
    'batches_for_validation' : 1000,
    'epochs' : 20,
    'max_trials' : 8,
    'executions_per_trial' : 1,
    'objective' : 'val_loss',
    'callbacks' : [es_callback, lr_callback, mc_callback]
    }

# params
hyperparameters = SmallImagesHP(
    init_min = 32, 
    init_max = 64, 
    cnn_layers_min = 2,
    cnn_layers_max = 2,
    cnn_min = 128,
    cnn_max = 192,
    cnn2_layers_min = 2,
    cnn2_layers_max = 2,
    cnn2_min = 64,
    cnn2_max = 128,
    dense_min = 400,
    dense_max = 800,
    dense_step = 50,
    dense2_min = 50,
    dense2_max = 250,
    dense2_step = 25,
    dropout = [0.5],
    learning_rate = [0.001, 0.0005]
    )

hp = SmallImagesModel.define_hp(hyperparameters)
hypermodel = SmallImagesModel(num_classes = 7, input_shape = (48, 48, 1))

run_tuner(hypermodel, hp)
input("Press Enter to rename files")
FileManager.rename_files(TUNER_SETTINGS['log_dir'], hypermodel.generate_model_name)