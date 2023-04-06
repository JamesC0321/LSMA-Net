from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D
from keras.layers import Activation, BatchNormalization, Add, Reshape, Multiply
from keras.layers import Lambda, Concatenate

import tensorflow as tf

from keras import backend as K
import numpy as np
import math


def slices(dw, n, data_format='channels_last'):
    if data_format == 'channels_last':
        return dw[:, :, :, :n]
    else:
        return dw[:, :n, :, :]

def _conv_block(inputs, outputs, kernel, strides, padding='same',
                use_relu=True, use_bias=False, data_format='channels_last'):

    channel_axis = -1 if K.image_data_format()=='channels_last' else 1
    x = Conv2D(outputs, kernel, padding=padding, strides=strides, use_bias=use_bias)(inputs)
    x = BatchNormalization(axis=channel_axis)(x)
    if use_relu:
        x = Activation('relu')(x)
    return x

def _squeeze(inputs, exp, ratio, data_format='channels_last'):

    input_channels = int(inputs.shape[-1]) if K.image_data_format() == 'channels_last' else int(inputs.shape[1])
    x = GlobalAveragePooling2D()(inputs)
    x = Reshape((1, 1, input_channels))(x)
    x = Conv2D(math.ceil(exp / ratio), (1, 1), strides=(1, 1), padding='same',data_format=data_format, use_bias=False)(x)
    x = Activation('relu')(x)
    x = Conv2D(exp, (1, 1), strides=(1, 1), padding='same',data_format=data_format, use_bias=False)(x)
    x = Activation('hard_sigmoid')(x)
    x = Multiply()([inputs, x])
    return x

def _ghost_module(inputs, exp, kernel, dw_kernel, ratio, s=1,padding='SAME',use_bias=False, data_format='channels_last',activation=None):

    output_channels = math.ceil(exp * 1.0 / ratio)
    x = Conv2D(output_channels, kernel, strides=(s, s), padding=padding,activation=activation, data_format=data_format,use_bias=use_bias)(inputs)
    if ratio == 1:
        return x
    dw = DepthwiseConv2D(dw_kernel, s, padding=padding, depth_multiplier=ratio-1,activation=activation,use_bias=use_bias)(x)
    dw = Lambda(slices,arguments={'n':exp-output_channels,'data_format':data_format})(dw)
    x = Concatenate(axis=-1 if data_format=='channels_last' else 1)([x,dw])

    return x

def _ghost_bottleneck(inputs, outputs, kernel, dw_kernel,exp, s, ratio, squeeze, name=None):

    data_format = K.image_data_format()
    channel_axis = -1 if data_format == 'channels_last' else 1
    input_shape = K.int_shape(inputs)  # 获取输入张量的尺寸
    # 步长为1 且 输入通道数=输出通道数
    # print("步长为1 且 输入通道数=输出通道数:",s == 1 and input_shape[channel_axis] == outputs)
    if s == 1 and input_shape[channel_axis] == outputs:
        res = inputs
    else:
        res = DepthwiseConv2D(kernel, strides=s, padding='SAME', depth_multiplier=ratio - 1,
                              data_format=data_format, activation=None, use_bias=False)(inputs)
        res = BatchNormalization(axis=channel_axis)(res)
        res = _conv_block(res, outputs, (1, 1), (1, 1), padding='valid',
                               use_relu=False, use_bias=False, data_format=data_format)
    x = _ghost_module(inputs, exp, [1, 1], dw_kernel, ratio)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    if s > 1:
        x = DepthwiseConv2D(dw_kernel, s, padding='same', depth_multiplier=ratio - 1,data_format=data_format, activation=None, use_bias=False)(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = Activation('relu')(x)
    if squeeze:
        '''oup = int(input_shape[channel_axis] * exp/outputs * 1.0)
        oup = round(max(oup, 8)/4)*4
        print(oup)'''
        x = _squeeze(x, exp, 4, data_format=data_format)
    x = _ghost_module(x, outputs, [1, 1], dw_kernel, ratio)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Add()([res, x])
    return x
