import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, ZeroPadding3D, Dense, Dropout, Activation, Convolution3D, Reshape
from tensorflow.keras.layers import AveragePooling3D, GlobalAveragePooling3D, MaxPooling3D, BatchNormalization


def densenet_model(growth_rate=6, nb_filter=6, nb_layers = [2,2], reduction=0.0, 
                   dropout_rate=0.3, classes=1, shape=(15,7,7,512), batch_size=64,
                   with_output_block=True, with_se_layers=True):
    # compute compression factor
    compression = 1.0 - reduction

    nb_dense_block = len(nb_layers)
    # From architecture for ImageNet (Table 1 in the paper)
    # nb_filter = 64
    # nb_layers = [6,12,24,16] # For DenseNet-121
    
    img_input = Input(shape=shape, name='data')
    
    x = ZeroPadding3D((3, 3,3), name='conv1_zeropadding', batch_size=batch_size)(img_input)
    x = Convolution3D(nb_filter, 7, 2, name='conv1', use_bias=False)(x)
    x = BatchNormalization(name='conv1_bn')(x)
    x = Activation('relu', name='relu1')(x)
    x = ZeroPadding3D((1, 1,1), name='pool1_zeropadding')(x)
    x = MaxPooling3D((3, 3,3), strides=(2, 2,2), name='pool1')(x)
    
    stage = 0
    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        stage = block_idx+2
        x, nb_filter = dense_block(x, stage, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=dropout_rate)

        if with_se_layers:
            x = se_block(x, stage, 'dense', nb_filter)

        # Add transition_block
        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate)
        nb_filter = int(nb_filter * compression)

        if with_se_layers:
            x = se_block(x, stage, 'transition', nb_filter)

    final_stage = stage + 1
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate, dropout_rate=dropout_rate)

    if with_se_layers:
        x = se_block(x, final_stage, 'dense', nb_filter)

    x = BatchNormalization(name='conv_final_blk_bn')(x)
    x = Activation('relu', name='relu_final_blk')(x)
    
    if not with_output_block:
        return Model(inputs=img_input, outputs=x)

    x = GlobalAveragePooling3D(name='pool_final')(x)
    x = Dense(classes, name='fc6')(x)
    #output = Activation('softmax', name='prob')(x)
    output = Activation('sigmoid', name='prob')(x) #changed to sigmoid because only 1 class
    return Model(inputs=img_input, outputs=output)

def conv_block(x, stage, branch, nb_filter, dropout_rate=None):
    conv_name_base = 'conv' + str(stage) + '_' + str(branch)
    relu_name_base = 'relu' + str(stage) + '_' + str(branch)

    # 1x1 Convolution (Bottleneck layer)
    inter_channel = nb_filter * 4  
    x = BatchNormalization(name=conv_name_base+'_x1_bn')(x)
    x = Activation('relu', name=relu_name_base+'_x1')(x)
    x = Convolution3D(inter_channel, 1, 1, name=conv_name_base+'_x1', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    # 3x3 Convolution
    x = BatchNormalization(name=conv_name_base+'_x2_bn')(x)
    x = Activation('relu', name=relu_name_base+'_x2')(x)
    x = ZeroPadding3D((1, 1, 1), name=conv_name_base+'_x2_zeropadding')(x)
    x = Convolution3D(nb_filter, 3, 1, name=conv_name_base+'_x2', use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x

def se_block(x, stage, previous, nb_filter, ratio = 16):
    se_name = 'se' + str(stage) + '_' + previous
    init = x
    x = GlobalAveragePooling3D(name='global_average_pooling_2d_'+se_name)(x)
    x = Dense(nb_filter // ratio, name='dense_relu_'+se_name)(x)
    x = Activation('relu', name='relu_'+se_name)(x)
    x = Dense(nb_filter, name='dense_sigmoid_'+se_name)(x)
    x = Activation('sigmoid', name='sigmoid_'+se_name)(x)
    x = tf.expand_dims(x,1)
    x = tf.expand_dims(x,1)
    x = init * tf.expand_dims(x,1) 
    return x

def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None, 
                grow_nb_filters=True):
    concat_feat = x
    for i in range(nb_layers):
        branch = i+1
        x = conv_block(concat_feat, stage, branch, growth_rate, dropout_rate)
        concat_feat = tf.concat([concat_feat, x], -1)

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter

def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None):
    conv_name_base = 'conv' + str(stage) + '_blk'
    relu_name_base = 'relu' + str(stage) + '_blk'
    pool_name_base = 'pool' + str(stage) 

    x = BatchNormalization(name=conv_name_base+'_bn')(x)
    x = Activation('relu', name=relu_name_base)(x)
    x = Convolution3D(int(nb_filter * compression), 1, 1, name=conv_name_base, use_bias=False)(x)

    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling3D((1, 2,2), strides=(2, 2,2), name=pool_name_base)(x)

    return x