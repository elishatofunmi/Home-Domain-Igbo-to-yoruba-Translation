import os, sys
import numpy as np

import pickle
#os.chdir(os.getcwd() +'/models/dtr')

input_dict = pickle.load(open('dict_input_char.pkl', 'rb'))
output_dict = pickle.load(open('dict_output_char.pkl','rb'))
lr0 = pickle.load(open('lr0dtr.pkl', 'rb'))
lr1 = pickle.load(open('lr1dtr.pkl', 'rb'))
lr2 = pickle.load(open('lr2dtr.pkl', 'rb'))
lr3 = pickle.load(open('lr3dtr.pkl', 'rb'))
lr4 = pickle.load(open('lr4dtr.pkl', 'rb'))
lr5 = pickle.load(open('lr5dtr.pkl', 'rb'))
lr6 = pickle.load(open('lr6dtr.pkl', 'rb'))


def encode_input_data(data):
    data = data.split()
    
    data_length = len(data)
    if data_length <5:
        diff = 5 - data_length
        for k in range(diff):
            data.append(' ')
    for i,j in enumerate(data):
        data[i] = dict_input_char[j.lower()]
    return data

def predict_encoder(data):
    data = np.array(data).reshape(1,-1)
    p0 = lr0.predict(data)
    p1 = lr1.predict(data)
    p2 = lr2.predict(data)
    p3 = lr3.predict(data)
    p4 = lr4.predict(data)
    p5 = lr5.predict(data)
    p6 = lr6.predict(data)
    return [float(p0),float(p1),float(p2),float(p3),float(p4),float(p5),float(p6)]


def three_dp(val):
    val_str = str(val)
    val_length = len(val_str)
    if val_length <5:
        diff = 5 - val_length
        for k in range(diff):
            val_str = val_str +'0'
        cut_val = val_str
    else:
        cut_val = val_str[:5]
    return float(cut_val)


def get_value(value):
    va = three_dp(0.0541/2)
    upper_range, lower_range = value +va, value - va
    key = ''
    for i,k in zip(list(dict_output_char.keys()), dict_output_char.values()):
        if k < three_dp(value) and k > lower_range:
            key = i
        elif k > three_dp(value) and k < upper_range:
            key = i
        else:
            continue
    return key

def model_decoder(data):
    result = []
    for k in data:
        result.append(get_value(k))
    output = ''
    for k in result:
        output += ' ' +k
    return output



def main(string):
    try:
        val = ''
        encoded_input_data = encode_input_data(string)
        prediction = predict_encoder(encoded_input_data)
        val = model_decoder(prediction)
    except KeyError as e:
        val = 'can\'t interpret this'
    return val


if __name__== '__main__':
    string = 'I saw it'
    result = main(string)
    print(result)