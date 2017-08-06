from keras.layers import Input,Dense
from keras.models import Model

"""
Simple autoencoder with Mnist dataset, can be easily changed to any other dataset
"""

def create_model(num_input,encoding_dims):
    """ Using functional API from keras """
    input_img = Input(shape=(num_input,))
    output = None

    prev_output=input_img
    for embedd in encoding_dims:
        output = Dense(embedd,activation='relu')(prev_output)
        prev_output=output
    encoder = Model(input_img,output)
    for embedd in encoding_dims[::-1][1:]:
        output = Dense(embedd,activation='relu')(prev_output)
        prev_output=output
    decoded = Dense(num_input,activation='sigmoid')(output)
    autoencoder = Model(input_img,decoded)

    autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')

    # Lets define a decoder for decoding after the model is trained
    """
    #This part created a new graph with new weights but is not trained when we train autoencoder
    encoded_input = Input(shape=(encoding_dims[-1],))
    prev_output_decoder=encoded_input
    output_decoder=None
    for embedd in encoding_dims[::-1][1:]:
        output_decoder = Dense(embedd,activation='relu')(prev_output_decoder)
        prev_output_decoder=output_decoder
    decoder_output = Dense(num_input,activation='sigmoid')(prev_output_decoder)
    decoder = Model(encoded_input,decoder_output)
    """

    encoded_input = Input(shape=(encoding_dims[-1],))
    index = -1 * len(encoding_dims)
    decoder_layer = autoencoder.layers[index]
    decoder = Model(encoded_input,decoder_layer(encoded_input))

    dict = {}
    dict['autoencoder'] = autoencoder
    dict['encoder'] = encoder
    dict['decoder'] = decoder
    return dict

if __name__ == '__main__':
    print(create_model(784,(392,196,98)))
