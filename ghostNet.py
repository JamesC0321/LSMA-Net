from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.layers.advanced_activations import LeakyReLU
import math
from keras import backend as K


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred= tf.cast(y_pred, dtype=tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1score(actual, prediction):
    prec = precision(actual, prediction)
    rec = recall(actual, prediction)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))

def slices(dw, n, data_format='channels_last'):
    if data_format == 'channels_last':
        return dw[:, :, :, :n]
    else:
        return dw[:, :n, :, :]

def coord_act(x,D=3):
    return x * ReLU(6.,)(x+3)/6

def CoordAtt1(x):

    x_h = pooling.AveragePooling2D(pool_size=(1, K.int_shape(x)[2]), strides=1, padding='valid')(x)
    x_w = pooling.AveragePooling2D(pool_size=(K.int_shape(x)[1], 1), strides=1, padding='valid')(x)
    x_w = Lambda(tf.transpose,arguments={'perm':[0, 2, 1, 3]})(x_w)
    y = concatenate([x_h, x_w], axis=1)
    y = Conv2D(8, (1, 1), strides=1, padding='valid')(y)
    y = BatchNormalization()(y)
    y = Lambda(coord_act,arguments={'D': 3 })(y)
    x_h, x_w = Lambda(tf.split,arguments={'num_or_size_splits':2,'axis':1})(y)
    x_w = Lambda(tf.transpose, arguments={'perm': [0, 2, 1, 3]})(x_w)
    a_h = Conv2D(K.int_shape(x)[3], (1, 1), strides=1, padding='valid',  activation='sigmoid',)(x_h)
    a_w = Conv2D(K.int_shape(x)[3], (1, 1), strides=1, padding='valid',  activation='sigmoid',)(x_w)
    return [a_h,a_w]

def my_upsampling(x,img_w,img_h,method=1):
  """0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法"""
  return tf.image.resize_images(x,(img_w,img_h),method)

def _squeeze(inputs, outputs, ratio, data_format='channels_last'):
    input_channels = int(inputs.shape[-1]) if K.image_data_format() == 'channels_last' else int(inputs.shape[1])
    x = GlobalAveragePooling2D()(inputs)
    x = Reshape((1, 1, input_channels))(x)
    x = Conv2D(math.ceil(outputs / ratio), (1, 1), strides=(1, 1), padding='same',data_format=data_format, use_bias=False)(x)
    x = Activation('relu')(x)
    x = Conv2D(outputs, (1, 1), strides=(1, 1), padding='same',data_format=data_format, use_bias=False)(x)
    x = Activation('hard_sigmoid')(x)
    x = Multiply()([inputs, x])
    return x

def AG2(x,high_f,inter_shape):

    low = Conv2D(inter_shape, 2, strides=2, padding='same')(x)
    high_f1 = Conv2D(inter_shape, 1, strides=1, padding='same', )(high_f)
    concat_xg = Add()([high_f1, low])
    concat_xg = Conv2D(inter_shape, 1,strides=1, padding='same',)(concat_xg)
    concat_xg = BatchNormalization()(concat_xg)
    y= CoordAtt2(concat_xg)
    low_f = Multiply()([y[0], x, y[1]])
    return low_f

def CoordAtt2(x):

    x_h = pooling.AveragePooling2D(pool_size=(1, K.int_shape(x)[2]), strides=1, padding='valid')(x)
    x_w = pooling.AveragePooling2D(pool_size=(K.int_shape(x)[1], 1), strides=1, padding='valid')(x)
    x_w =Lambda(tf.transpose,arguments={'perm':[0, 2, 1, 3]})(x_w)
    y = concatenate([x_h, x_w], axis=1)
    y = Conv2D(8, (1, 1), strides=1, padding='valid')(y)
    y = BatchNormalization()(y)
    y = Lambda(coord_act, arguments={'D': 3})(y)
    x_h, x_w = Lambda(tf.split,arguments={'num_or_size_splits':2,'axis':1})(y)
    x_w = Lambda(tf.transpose, arguments={'perm': [0, 2, 1, 3]})(x_w)
    a_h = Conv2D(K.int_shape(x)[3], (1, 1), strides=1, padding='valid',  activation='sigmoid',)(x_h)
    a_w = Conv2D(K.int_shape(x)[3], (1, 1), strides=1, padding='valid',  activation='sigmoid',)(x_w)
    a_h = Lambda(my_upsampling,arguments={'img_w':2*K.int_shape(x)[2],'img_h':1})(a_h)
    a_w = Lambda(my_upsampling,arguments={'img_w':1,'img_h':2*K.int_shape(x)[1]})(a_w)
    return [a_h,a_w]

def adapt_ca(inputs,outputs):
    conv1 = Conv2D(outputs, 1, strides=1, padding='same',)(inputs)
    conv2 = Conv2D(outputs, 3, strides=1, padding='same', dilation_rate=3)(inputs)
    conv3 = Conv2D(outputs, 3, strides=1, padding='same', dilation_rate=5)(inputs)
    conv4 = Add()([conv1, conv2,conv3])
    conv4 = Conv2D(outputs, 3, strides=1, padding='same',kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = ReLU(max_value=6)(conv4)
    conv4 = Conv2D(outputs, 1, strides=1, padding='same',kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    y= CoordAtt1(conv4)
    new_conv1 = Multiply()([y[0], conv1, y[1]])
    new_conv2 = Multiply()([y[0], conv2, y[1]])
    new_conv3 = Multiply()([y[0], conv3, y[1]])
    out = Add()([new_conv1, new_conv2,new_conv3,inputs])
    out = Conv2D(outputs, 1, strides=1,padding='same',kernel_initializer='he_normal')(out)
    out = BatchNormalization()(out)
    return out

def _conv_block(inputs, outputs, kernel, strides, padding='same',use_relu=True, use_bias=False, data_format='channels_last'):
    channel_axis = -1 if K.image_data_format()=='channels_last' else 1
    x = Conv2D(outputs, kernel, padding=padding, strides=strides, use_bias=use_bias, data_format=data_format,kernel_initializer='he_normal')(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    if use_relu:
        x = ReLU(max_value=6)(x)
    return x


def _ghost_module(inputs, outputs, kernel, dw_kernel, ratio, s=1,padding='SAME',use_bias=False, data_format='channels_last',use_relu=False):
    channel_axis = -1 if data_format == 'channels_last' else 1
    output_channels = math.ceil(outputs * 1.0 / ratio)
    x = Conv2D(output_channels, kernel, strides=(s, s), padding=padding, data_format=data_format,use_bias=use_bias, kernel_initializer='he_normal')(inputs)
    # x = BatchNormalization(axis=channel_axis)(x)
    if use_relu:
        x = LeakyReLU(alpha=0.3)(x)
    if ratio == 1:
        return x
    dw = DepthwiseConv2D(dw_kernel, s, padding=padding, depth_multiplier=ratio-1,use_bias=use_bias)(x)
    # dw = BatchNormalization(axis=channel_axis)(dw)
    if use_relu:
        dw = LeakyReLU(alpha=0.3)(dw)
    dw = Lambda(slices,arguments={'n':outputs-output_channels,'data_format':data_format})(dw)
    x = Concatenate(axis=-1 if data_format=='channels_last' else 1)([x,dw])
    return x

def fuse(x_hight,x_low,k):

    x = Conv2D(k, 3, strides=(1, 1), padding='same',  kernel_initializer='he_normal')(x_hight)
    mid = Multiply()([x, x_low])
    mid = Conv2D(k, 3, strides=(1, 1), padding='same',  kernel_initializer='he_normal')(mid)

    out = Add()([mid, x])
    return out

def BasicBlock(inputs, out_channels,expansion_factor=2,dw_kernel=3, ratio=2, strides=1,):
    data_format = K.image_data_format()
    channel_axis = -1 if data_format == 'channels_last' else 1

    mid_channels = out_channels//expansion_factor
    conv1 = _ghost_module(inputs,out_channels,kernel=3,dw_kernel=dw_kernel, ratio=ratio, s=1,use_relu=True)
    conv1 = BatchNormalization(axis=channel_axis)(conv1)

    conv2 =_conv_block(conv1, mid_channels, 1, 1, padding='same',use_relu=False)
    conv3 = _conv_block(conv2, out_channels, 1, 1, padding='same', use_relu=True)

    conv4 = _ghost_module(conv3,out_channels,kernel=3,dw_kernel=dw_kernel, ratio=ratio, s=strides,use_relu=False)
    conv4 = BatchNormalization(axis=channel_axis)(conv4)

    shortcut =  _ghost_module(inputs,out_channels,kernel=1,dw_kernel=dw_kernel, ratio=ratio, s=strides,use_relu=False)
    shortcut = BatchNormalization(axis=channel_axis)(shortcut)

    out = Add()([conv4, shortcut])
    return out

ratio = 2
dw_kernel = 3

def unet(pretrained_weights=None, input_size=(512, 512, 3)):
    inputs = Input(input_size)


    conv1 =_conv_block(inputs, 16, 3, 1, padding='same', use_relu=False)
    conv1 = ReLU()(conv1)
    conv1 = BasicBlock(conv1, 16, expansion_factor=2,dw_kernel=3, ratio=2, strides=1)
    conv11 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = BasicBlock(conv11, 48,expansion_factor=4, dw_kernel=3, ratio=2, strides=1)
    conv22 = MaxPooling2D(pool_size=(2, 2))(conv2)


    conv3 = BasicBlock(conv22, 64,expansion_factor=4, dw_kernel=3, ratio=2, strides=1)
    conv33 = MaxPooling2D(pool_size=(2, 2))(conv3)


    conv4 = BasicBlock(conv33, 96,expansion_factor=4, dw_kernel=3, ratio=2, strides=1)
    drop4 = Dropout(0.5)(conv4)
    conv44 = MaxPooling2D(pool_size=(2, 2))(drop4)


    conv5 = BasicBlock(conv44, 160,expansion_factor=4, dw_kernel=3, ratio=2, strides=1)
    drop5 = Dropout(0.5)(conv5)

    oo = adapt_ca(drop5,160)

    up6 = Conv2D(96, 2, padding='same', activation='relu', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(oo))
    ag1 = AG2(drop4, oo, 96)
    merge6 = Add()([ag1, up6])
    conv6 = Conv2D(96, 3, padding='same', activation='relu', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(96, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(64, 2, padding='same', activation='relu', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))
    ag2 = AG2(conv3, conv6, 64)
    merge7 = Add()([ag2, up7])
    conv7 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(64, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(48, 2, padding='same', activation='relu', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))
    ag3 = AG2(conv2, conv7, 48)
    merge8 = Add()([ag3, up8])
    conv8 = Conv2D(48, 3, padding='same', activation='relu', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(48, 3, padding='same', activation='relu', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(16, 2, padding='same', activation='relu', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))
    ag4 = AG2(conv1, conv8, 16)
    merge9 = Add()([ag4, up9])
    merge9 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(merge9)
    merge9 = Conv2D(16, 3, padding='same', activation='relu', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(2, 3, padding='same', activation='relu', kernel_initializer='he_normal')(merge9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(lr=1e-3),loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def Conv_bn_ac(inputs,outputs,k=3,activation = 'relu',padding = 'same',kernel_initializer = 'he_normal'):

    conv1 = Conv2D(outputs, k, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(outputs, k, activation=activation, padding=padding, kernel_initializer=kernel_initializer)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    return conv2
