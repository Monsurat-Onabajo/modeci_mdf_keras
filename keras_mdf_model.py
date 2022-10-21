
# This is a continuation of the modeci_mdf to keras project, this is the number 4 task https://github.com/ModECI/MDF/discussions/345

import numpy as np
from tensorflow.keras.models import load_model
from modeci_mdf.execution_engine import EvaluableGraph
from modeci_mdf.utils import simple_connect
from modeci_mdf.mdf import *

# this python script loads keras model that was trained in the last jupyter notebook and parses the structure and then the structure was exported to the eqiovalent network in mdf

def keras_model():
    """
    this function returns the trained keras model
    parameters
    ----------
    None
    """
    model= load_model('model.h5')
    return model


def weight_and_activation(model):
    """
    this function takes in a trained keras model and return the activations,
    weight and bias for each layer in the model
   
    parameters:
    model: trained keras model
    """
    params= {}
    activation=[]
    layers_of_model= ['dense', 'dense_1', 'dense_2']
    for item in layers_of_model:
        dic= {}
        layer= model.get_layer(item)
        weight,bias = layer.weights
        dic['weights']= np.array(weight)
        dic['bias']= np.array(bias)
        params[item]= dic
        activation.append(str(layer.activation).split()[1])
    return params, activation


params, activation= weight_and_activation(model= keras_model())


def converting_keras_mdf(input_array):
    """
    this function takes in an input array and returns the equivalent predictions
    for this model, we are using an array of 0s and ones to predict a three input XOR
    truth table where x= np.array([0,0,0]) and y= 1,  x_1= np.array([1,0,0]) and y_1= 0,
    x_2= np.array([0,0,0]) and y_2= 1
    
    this function will return the modeci_mdf model equivalent of the keras model
    parameters
    ----------
    input_array: an array of three columns of 0 and 1 in any particular order,
    dtype should be float
    e.g np.array([[1,0,1], [0,0,1], [0,0,0]]), dtype= float)
    """
    # creating model and appending to graph
    model_mdf= Model(id= 'XOR_project')
    graph= Graph(id= 'Implementing Keras in MDF')
    model_mdf.graphs.append(graph)

    # create input layer node

    input_layer= Node(id= 'input layer')
    input_layer.parameters.append(Parameter(id= 'input_layer', value=input_array))
    input_layer.output_ports.append(OutputPort(id= 'output', value= 'input_layer'))
    graph.nodes.append(input_layer)


    # create first layer node, append to graph and connect it to input node
    first_node= Node(id='dense_layer', metadata= {'color':'.8 .8 .8'})
    first_node.input_ports.append(InputPort(id= 'input'))
    first_node.parameters.append(Parameter(id='dense_weight', value= params['dense']['weights']))
    first_node.parameters.append(Parameter(id='dense_bias', value= params['dense']['bias']))
    feedForward= Parameter(id= 'feedForward',
                           value= '(input @ dense_weight) + dense_bias')
    first_node.parameters.append(feedForward)
    first_node.output_ports.append(OutputPort(id= 'dense_output', value= 'feedForward'))
    graph.nodes.append(first_node)
    simple_connect(input_layer, first_node, graph)

    # create first layer activation,append to graph and connect it to first layer node
    activation_1= Node(id= 'dense_activation')
    activation_1.input_ports.append(InputPort(id='input'))
    activation= Parameter(id= 'relu',
                      value = 'input*input')
    
    activation.conditions.append(ParameterCondition(id= 'test',
                            test= 'relu > 0', value = 'relu'))

    activation_1.parameters.append(activation)
    activation_1.output_ports.append(OutputPort(id='activation_1', value= 'relu' ))
    graph.nodes.append(activation_1)
    simple_connect(first_node, activation_1, graph)
    
    
    # create second layer node, append to graph and connect it to first layer activation node
    second_node= Node(id='dense_1_layer', metadata= {'color':'.8 .8 .8'})
    second_node.input_ports.append(InputPort(id= 'input'))
    second_node.parameters.append(Parameter(id='dense_1_weight', value= params['dense_1']['weights']))
    second_node.parameters.append(Parameter(id='dense_1_bias', value= params['dense_1']['bias']))
    feedForward= Parameter(id= 'feedForward',
                           value= '(input @ dense_1_weight) + dense_1_bias')
    second_node.parameters.append(feedForward)

    second_node.output_ports.append(OutputPort(id= 'dense_1_output', value= 'feedForward'))

    graph.nodes.append(second_node)

    simple_connect(activation_1, second_node, graph)


    # create second layer activation node, append to graph and connect it to second layer node
    activation_2= Node(id= 'dense_1_activation')
    activation_2.input_ports.append(InputPort(id='input'))
    activation= Parameter(id= 'relu',
                      value = 'input*input')
    activation.conditions.append(ParameterCondition(id= 'test',
                            test= 'relu > 0', value = 'relu'))

    activation_2.parameters.append(activation)
    activation_2.output_ports.append(OutputPort(id='activation_2', value= 'relu' ))
    graph.nodes.append(activation_2)
    simple_connect(second_node,activation_2, graph)

    # create third layer node, append to graph and connect it to second layer activation node

    third_node= Node(id='dense_2_layer', metadata= {'color':'.8 .8 .8'})
    third_node.input_ports.append(InputPort(id= 'input'))
    third_node.parameters.append(Parameter(id='dense_2_weight', value= params['dense_2']['weights']))
    third_node.parameters.append(Parameter(id='dense_2_bias', value= params['dense_2']['bias']))
    feedForward= Parameter(id= 'feedForward',
                           value= '(input @ dense_2_weight) + dense_2_bias')
    third_node.parameters.append(feedForward)

    third_node.output_ports.append(OutputPort(id= 'dense_2_output', value= 'feedForward'))
    graph.nodes.append(third_node)
    simple_connect(activation_2, third_node, graph)
    
    # create third layer activation node, append to graph and connect it to third layer node

    # 2.71828 is mathematics constant for exponential function

    activation_3= Node(id= 'dense_2_activation')
    activation_3.input_ports.append(InputPort(id='input'))
    activation= Parameter(id= 'sigmoid', 
                         value= '1/(1 + (2.71828**(-input)))' )
    activation_3.parameters.append(activation)
    activation_3.output_ports.append(OutputPort(id='activation_2', value= 'sigmoid' ))
    graph.nodes.append(activation_3)
    simple_connect(third_node, activation_3, graph)

    # return keras model that has been converted to mdf model and its equivalent graph
    
    return model_mdf, graph






