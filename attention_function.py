import numpy as np
import pandas as pd
import keras.backend as K
import keras
from keras.layers import multiply
from keras.layers.core import Dense, Reshape, Lambda, RepeatVector, Permute, Flatten
from keras.layers.recurrent import LSTM
from keras.models import Model, Input
from keras import regularizers
# plot part.
import matplotlib.pyplot as plt


# ## Helper functions

def get_activations(model, inputs, print_shape_only=False, layer_name=None, verbose=False):
    """
    Get activations from a model
    Args:
        model: a keras model
        inputs: the inputs for the model
        print_shape_only: whether to print the shape of the layer or the whole activation layer
        layer_name: name of specific layer to return
        verbose: whether to show all outputs
    Returns:
        activations: list, list of activations
    """
    activations = []
    inp = model.input
    if layer_name is None:
        outputs = [layer.output for layer in model.layers]
    else:
        outputs = [layer.output for layer in model.layers if layer.name == layer_name]  # all layer outputs
    funcs = [K.function([inp] + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    layer_outputs = [func([inputs, 1.])[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if verbose:
            print('----- activations -----')
            if print_shape_only:
                print(layer_activations.shape)
            else:
                print(layer_activations)
    return activations



def get_data_recurrent(n, time_steps, input_dim, attention_column=10):
    """
    Data generation. x is random except that first value equals the target y.
    network should learn that the target = x[attention_column].
    Therefore, most of its attention should be focused on the value addressed by attention_column.
    Args:
        n: the number of samples to retrieve.
        time_steps: the number of time steps of your series.
        input_dim: the number of dimensions of each element in the series.
        attention_column: the column linked to the target. Everything else is purely random.
    Returns:
        x: model inputs
        y: model targets
    """
    x = np.random.standard_normal(size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y


def attention_3d_block(inputs, TIME_STEPS): #attn time 
    """
    inputs.shape = (batch_size, time_steps, input_dim)
    """ 
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a)
    a = Dense(TIME_STEPS, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

def attention_time_reg(inputs, TIME_STEPS,kreg,areg): #attn time_reg
    """
    inputs.shape = (batch_size, time_steps, input_dim)
    """
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a)
    a = Dense(TIME_STEPS, activation='softmax',kernel_regularizer=regularizers.l2(kreg),activity_regularizer = regularizers.l2(areg))(a) #0.01
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

def attention_time_reg_l1(inputs, TIME_STEPS,kreg,areg): #attn time_reg
    """
    inputs.shape = (batch_size, time_steps, input_dim)
    """
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, TIME_STEPS))(a)
    a = Dense(TIME_STEPS, activation='softmax',kernel_regularizer=regularizers.l1(kreg),activity_regularizer = regularizers.l1(areg))(a) #0.01
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul


def attention_3d_block_time_features(inputs, TIME_STEPS): #attn both
    """
    inputs.shape = (batch_size, time_steps, input_dim)
    """ 
    input_dim = int(inputs.shape[2])
    a = Flatten()(inputs)
    a = Dense(TIME_STEPS*input_dim, activation='softmax')(a)
    a = Reshape((input_dim, TIME_STEPS))(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

def attention_both_reg(inputs, TIME_STEPS,kreg,areg): #attn both reg
    """
    inputs.shape = (batch_size, time_steps, input_dim)
    """
    input_dim = int(inputs.shape[2])
    a = Flatten()(inputs)
    a = Dense(TIME_STEPS*input_dim, activation='softmax',kernel_regularizer=regularizers.l2(kreg), activity_regularizer=regularizers.l2(areg))(a)
    a = Reshape((input_dim, TIME_STEPS))(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul


def attention_spatial_block(inputs,TIME_STEPS): #attn feat
    """
    inputs.shape = (batch_size, time_steps, input_dim)
    """ 
    input_dim = int(inputs.shape[2])
    a = Reshape((TIME_STEPS, input_dim))(inputs)
    a_probs = Dense(input_dim, activation='softmax', name='attention_vec')(a)
    #output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    #output_attention_mul = keras.layers.Add(name='attention_mul')([inputs, a_probs])
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

def attention_feat_reg(inputs,TIME_STEPS,kreg,areg): #attn feat
    """
    inputs.shape = (batch_size, time_steps, input_dim)
    """
    input_dim = int(inputs.shape[2])
    a = Reshape((TIME_STEPS, input_dim))(inputs)
    a_probs = Dense(input_dim, activation='softmax', name='attention_vec',kernel_regularizer=regularizers.l2(kreg),activity_regularizer=regularizers.l2(areg))(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul

def attention_feat_reg_l1(inputs,TIME_STEPS,kreg,areg): #attn feat
    """
    inputs.shape = (batch_size, time_steps, input_dim)
    """
    input_dim = int(inputs.shape[2])
    a = Reshape((TIME_STEPS, input_dim))(inputs)
    a_probs = Dense(input_dim, activation='softmax', name='attention_vec',kernel_regularizer=regularizers.l1(kreg),activity_regularizer=regularizers.l1(areg))(a)
    output_attention_mul = multiply([inputs, a_probs])
    return output_attention_mul


# ## Hyperparameters and builder methods

def model_attention_applied_before_lstm():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    attention_mul = attention_3d_block(inputs)
    #attention_mul = attention_spatial_block(inputs)
    lstm_units = 32
    attention_mul = LSTM(lstm_units, return_sequences=False)(attention_mul)
    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(input=[inputs], output=output)
    return model

"""
def groupsparse(activity):
    features = ['tod','dow','tod_next','dow_next','gender','age','day since quit','income','curr_completed','longterm_rate',
                    'shortterm_rate','active','angry','ashamed','calm', 'determined', 'disgusted', 'enthusiastic', 'grateful',
                    'guilty','happy','irritable', 'lonely', 'proud', 'nervous', 'sad', 'restless','tired', 'hopeless', 'scared',
                    'bored','joyful', 'attentive','relaxed', 'motivation', 'urge','incentive','active_var','angry_var',
                    'ashamed_var','calm_var','determined_var','disgusted_var','enthusiastic_var','grateful_var','guilty_var',
                    'happy_var','irritable_var','lonely_var','proud_var','nervous_var','sad_var','restless_var','tired_var',
                    'hopeless_var','scared_var','bored_var','joyful_var','attentive_var','relaxed_var','motivation_var',
                    'urge_var','incentive_var']
    time_feat = ['tod','dow','tod_next','dow_next','day since quit']
    demo_feat = ['gender','age','income']
    comp_feat = ['curr_completed','longterm_rate','shortterm_rate','incentive','incentive_var']
    pos_feat = ['active','attentive','motivation','determined','calm','relaxed','enthusiastic','grateful','proud','joyful',
                'happy']
    neg_feat = ['angry','ashamed','disgusted','irritable', 'lonely','nervous', 'sad', 'restless','tired', 'hopeless',
                'scared', 'bored','guilty','urge']
    pos_var = ['active_var','attentive_var','motivation_var','determined_var','calm_var','relaxed_var','enthusiastic_var',
               'grateful_var','proud_var','joyful_var','happy_var']
    neg_var = ['angry_var','ashamed_var','disgusted_var','irritable_var','lonely_var','nervous_var','sad_var',
                   'restless_var','tired_var','hopeless_var','scared_var','bored_var','guilty_var','urge_var']

    ind_t = []
    for i in time_feat:
        ind_t.append(features.index(i))

    ind_t = tuple(slice(x) for x in ind_t)
    
    ind_d = []
    for i in demo_feat:
        ind_d.append(features.index(i))
    ind_d = tuple(slice(x) for x in ind_d)

    ind_c = []
    for i in comp_feat:
        ind_c.append(features.index(i))
    ind_c = tuple(slice(x) for x in ind_c)

    ind_pos = []
    for i in pos_feat:
        ind_pos.append(features.index(i))
    ind_pos = tuple(slice(x) for x in ind_pos)

    ind_neg = []
    for i in neg_feat:
        ind_neg.append(features.index(i))
    ind_neg = tuple(slice(x) for x in ind_neg)

    ind_pv = []
    for i in pos_var:
        ind_pv.append(features.index(i))
    ind_pv = tuple(slice(x) for x in ind_pv)
    
    ind_nv = []
    for i in neg_var:
        ind_nv.append(features.index(i))
    ind_nv = tuple(slice(x) for x in ind_nv)


    group_l2 = 0.001* (K.abs(K.sum(K.square(activity[:,:,ind_t]))) + K.abs(K.sum(K.square(activity[:,:,ind_d])))+ K.abs(K.sum(K.square(activity[:,:,ind_c])))+ K.abs(K.sum(K.square(activity[:,:,ind_pos])))+K.abs(K.sum(K.square(activity[:,:,ind_neg])))+K.abs(K.sum(K.square(activity[:,:,ind_pv])))+ K.abs(K.sum(K.square(activity[:,:,ind_nv]))))

    return group_l2


def attention_feat_groupsparse(inputs,TIME_STEPS):
    input_dim = int(inputs.shape[2])
    a = Reshape((TIME_STEPS, input_dim))(inputs)
    a_probs = Dense(input_dim,activation='softmax',name='attention_vec',activity_regularizer=groupsparse)(a)
    output_attention_mul = multiply([inputs,a_probs])
    return output_attention_mul

"""
