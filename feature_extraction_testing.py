


########################load libraries#########################################
import time
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras import applications
from keras.optimizers import SGD
import numpy as np
from densenet121_models import densenet121_model 

from keras.callbacks import ModelCheckpoint
    
#########################image characteristics#################################
img_rows=100 #dimensions of image
img_cols=100
channel = 3 #RGB
num_classes = 2 
batch_size = 1 #vary depending on the GPU
num_epoch = 60
###############################################################################
''' This code uses VGG-16 as a feature extractor'''

# create the base pre-trained model
#base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=(100,100,3))

#you can use the rest of the models like:
feature_model = applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
 
#feature_model = applications.Xception(weights='imagenet', include_top=False, input_shape=(100,100,3))
#For DenseNet, the main file densenet121_model is included to this repository.
#The model can be used as :
    
#feature_model = densenet121_model(img_rows=img_rows, img_cols=img_cols, color_type=channel, num_classes=num_classes)

#extract feature from an intermediate layer
#base_model = Model(input=base_model.input, output=base_model.get_layer('block5_conv2').output) 

#you can use the rest of the models like this:
feature_model = Model(input=feature_model.input, output=feature_model.get_layer('res5c_branch2c').output) #for ResNet50

#feature_model = Model(input=feature_model.input, output=feature_model.get_layer('block14_sepconv1').output) #for Xception
base_model= feature_model
#get the model summary
base_model.summary()
###############################################################################
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
# and a logistic layer 
predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
###############################################################################
# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers to prevent large gradient updates wrecking the learned weights
for layer in base_model.layers:
    layer.trainable = False
# compile the model (should be done *after* setting layers to non-trainable)
#fix the optimizer
sgd = SGD(lr=0.00001, decay=1e-6, momentum=0.9, nesterov=True) 
#compile the gpu model
model.compile(optimizer=sgd,
              loss='mse',
              metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_set = ImageDataGenerator()
val_set =  ImageDataGenerator()

train = train_set.flow_from_directory('cell_image/train',
                                                target_size = (100,100),
                                                #batch_size = 1)
                                                class_mode = 'categorical')

valid = val_set.flow_from_directory('cell_image/valid',
                                            target_size = (100, 100),
                                            #batch_size = 32,
                                            class_mode = 'categorical')
'''
###############################################################################
#load data for training
X_train, Y_train = load_resized_training_data(img_rows, img_cols)
X_valid, Y_valid = load_resized_validation_data(img_rows, img_cols)
#print the shape of the data
print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)
###############################################################################
'''
t=time.time() #make a note of the time
#start training
print('-'*30)
print('Start Training the model...')
print('-'*30)


filepath="resnet_log/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc',verbose=0,save_best_only=False, save_weights_only=False, mode='auto', period=1)
#from keras.callbacks import CSVLogger

#csv_logger = CSVLogger('training_6.log', append=True, separator=',')
callbacks_list = [checkpoint]

hist = model.fit_generator(train,
     
      epochs=num_epoch,
      shuffle=True,
      steps_per_epoch = 8000,
      verbose=1,
      validation_data = valid,
      validation_steps = 2000,
      #validation_split=0.2,
      callbacks=callbacks_list)

#model.save('my_model.h5') 
#print the history of the trained model
print(hist.history)

#compute the training time
print('Training time: %s' % (time.time()-t))
###############################################################################
# Make predictions on validation data
print('-'*30)
print('Predicting on validation data...')
print('-'*30)


#-------------------------------


from keras.models import load_model
model = load_model('vgg_log/weights-improvement-20-0.92.hdf5')


y_pred = model.predict(valid, batch_size=batch_size, verbose=1)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('test/test_uninfected.png', target_size = (100, 100))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
train.class_indices
if result[0][0] == 1:
    prediction = 'Parasitized'
else:
    prediction = 'Uninfected'

print(prediction)











