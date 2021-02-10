from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D, Dense
from keras.layers import Flatten, Dropout, Dense
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam

from kerastuner import HyperModel, HyperParameters

class SmallImagesModel(HyperModel):

    def __init__(self, num_classes, input_shape):
        self.num_classes = num_classes
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()

        #init layer 2 options 32/64
        model.add(Conv2D(filters = hp['init'], kernel_size = (3, 3), input_shape = self.input_shape))
        model.add(PReLU())
        model.add(MaxPooling2D((2, 2)))
        
        #test few cnn layers 3 -> 27 + 9 + 3  => 39 options
        for layer in range(1, hp['cnn_layers'] + 1):
            model.add(Conv2D(hp['cnn_{0}'.format(layer)], (3, 3)))
            model.add(PReLU())

        if(hp['cnn_layers'] > 1):
            model.add(MaxPooling2D((2, 2))) 

        model.add(Flatten())

        #dropout 2 options
        model.add(Dropout(hp['dropout']))
        #dense 2 options
        model.add(Dense(hp['dense']))
        model.add(PReLU())

        model.add(Dense(self.num_classes, activation='softmax'))
        
        #3 options
        opt = Adam(learning_rate=hp['learning_rate'])

        #model has 2 * 39 * 2 * 2 * 3 = 936 options
        model.compile(optimizer = opt)
        return model

    #default 936 options
    def define_hp(hp_model = None):
        hp = HyperParameters()
        
        if(hp_model is None):
            hp_model = SmallImagesHP()

        hp.Int(name = 'init',
            min_value = hp_model.init_min,
            max_value = hp_model.init_max,
            step = 32)  
                
        hp.Int(name = 'cnn_layers',
            min_value = hp_model.cnn_layers_min,
            max_value = hp_model.cnn_layers_max,
            step = 1)

        for i in range(1, hp_model.cnn_layers_max + 1):
            hp.Int(name = 'cnn_{0}'.format(i),
                min_value = hp_model.cnn_min,
                max_value = hp_model.cnn_max,
                step = 32)             

        hp.Int(name = 'dense',
            min_value = hp_model.dense_min,
            max_value = hp_model.dense_max,
            step = hp_model.dense_step)

        hp.Choice('dropout', hp_model.dropout)
        hp.Choice('learning_rate', hp_model.learning_rate)

        return hp

    def generate_model_name(iterable, **kwarg):
        hp = kwarg['hp']
        name = f"InitCNN({hp['init']})_Pool_"
        for layer in range(1, hp['cnn_layers'] + 1):
            name = f"{name}CNN({hp[f'cnn_{layer}']})_"

        if(hp['cnn_layers']> 1):
            name = f"{name}Pool_"

        name = f"{name}Dense({hp['dense']})_"
        name = f"{name}Dropout({hp['dropout']})_"
        name = f"{name}LR({hp['learning_rate']})"        
            
        return name

class SmallImagesHP():
    """!Important! Ensure that if min provided, then max > min is specified either.
    """
    def __init__(self,
                init_min = 32, 
                init_max = 64, 
                cnn_layers_min = 1,
                cnn_layers_max = 3,
                cnn_min = 32,
                cnn_max = 96,
                dense_min = 75,
                dense_max = 125,
                dense_step = 25,
                dropout = [0.2,0.3,0.5],
                learning_rate = [0.002, 0.001, 0.0005]
                ):
        self.init_min = init_min
        self.init_max = init_max
        
        self.cnn_layers_min = cnn_layers_min
        self.cnn_layers_max = cnn_layers_max      
        self.cnn_min = cnn_min
        self.cnn_max = cnn_max
        
        self.dense_min = dense_min
        self.dense_max = dense_max
        self.dense_step = dense_step
        
        self.dropout = dropout
        self.learning_rate = learning_rate
        