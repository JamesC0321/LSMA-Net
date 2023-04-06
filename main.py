from ghostNet import *
from data import *
import keras.backend as K
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint

def scheduler(epoch):
    if epoch % 15 == 0 and epoch != 0 or epoch >= 100 and epoch % 8 == 0 or epoch >= 300 and epoch % 4 == 0 or epoch >= 500 and epoch % 2 == 0:
    # if epoch % 50 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.9)
        print("lr changed to {}".format(lr * 0.9))
    return K.get_value(model.optimizer.lr)

data_gen_args = dict(rotation_range=60,
                    width_shift_range=0.01,
                    height_shift_range=0.01,
                    shear_range=0.01,
                    zoom_range=0.01,
                    vertical_flip=True,
                    horizontal_flip=True,#翻转
                    fill_mode='nearest',)


num = [3]
for i in num:
    myGene = trainGenerator(2,'data/membraneST/'+str(i)+'/train','image','label',data_gen_args,save_to_dir = None)
    valid =vaildgen('data/membraneST/'+str(i)+'/val/image/','data/membraneST/'+str(i)+'/val/label/')
    # valid =vdata(3,'data/membrane/train/valid','image','label',save_to_dir = None)
    model = unet(input_size=(512, 512, 3))
    # tbCallBack = TensorBoard(log_dir="./model")
    model_checkpoint = ModelCheckpoint('./data/membraneST/'+str(i)+'/model/unet_membrane.hdf5', monitor='val_loss',verbose=0, save_best_only=True)
    change_lr = LearningRateScheduler(scheduler)
    model.fit_generator(myGene,steps_per_epoch=300,epochs=30,validation_data=valid,verbose=1,callbacks=[model_checkpoint,change_lr])





