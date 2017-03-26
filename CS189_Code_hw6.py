#Neural Network Image Classifier
#For Computer Science 189 at UC Berkeley
#Dan March
#Spring 2016

import numpy as np
from scipy import io
import random
import math
import matplotlib.pyplot as plt

#Trains the Neural Network.
def trainNeuralNetwork(learning_rate=.01,cost_fn='squared',validate=True,\
                        test=False,plotting=False,epoch_number=20):
    number_of_iterations = 50000*epoch_number
    epoch = 20
    validation_set_size = 10000
    data = io.loadmat("dataset/train.mat")
    test_data = io.loadmat("dataset/test.mat")
    training_features = np.array(data['train_images'])
    training_labels = np.array(list(map(int,(data['train_labels']))))

    test_images = np.array(test_data['test_images'].T)

    training_features,training_labels = shuffle(training_features,training_labels)

    if validate:
        #Extracts the validation set:
        validation_set = []
        for i in range(validation_set_size):
            sample = [training_features[i],training_labels[i]]
            max_feature_value = max(sample[0])
            max_predicted_value = 9
            sample[0] = np.append(sample[0],[1])
            sample[0] = np.divide(sample[0],max_feature_value)
            sample[1] = sample[1]/max_predicted_value
            validation_set.append([sample[0],vectorize_training_label(sample[1])])

        print("Validation set constructed.")

    X_matrix = np.random.normal(scale=.01,size=(200,785))
    W_matrix = np.random.normal(scale=.01,size=(10,201))

    count = validation_set_size
    plot_x_values,plot_y_values = [],[]

    while count < number_of_iterations+validation_set_size:
        ind = count % epoch
        printed_iteration = count - validation_set_size
        if printed_iteration % 1000 == 0:
            print("Iteration " + str(printed_iteration))
            if plotting:
                if cost_fn == 'squared':
                    plt.xlabel('Iterations')
                    plt.ylabel('Training Error')
                    plt.title('Training Error v. SGD Iterations with Mean Squared Error')
                    plot_x_values.append(printed_iteration)
                    validation_error = validate_func(validation_set,W_matrix,X_matrix)
                    plot_y_values.append(validation_error)
                else:
                    plt.xlabel('Iterations')
                    plt.ylabel('Training Error')
                    plt.title('Training Error v. SGD Iterations with Cross-Entropy Error')
                    plot_x_values.append(printed_iteration)
                    validation_error = validate_func(validation_set,W_matrix,X_matrix)
                    print("validation error: " + str(validation_error))
                    plot_y_values.append(validation_error)

        #training_sample takes form: (flattened image vector,predicted number)
        training_sample = [training_features[ind],training_labels[ind]]
        #Preprocessing: Normalizes all features and predicted values. Also
        #appends 1 to the input training sample to model bias.
        max_feature_value = max(training_sample[0])
        max_predicted_value = 9
        training_sample[0] = np.append(training_sample[0],[1])
        training_sample[0] = np.divide(training_sample[0],max_feature_value)
        training_sample[1] = training_sample[1]/max_predicted_value
        #Forward propagation: Since this is a one-hidden-layer neural net,
        #this takes place in three steps.
        output = forward_propagation(X_matrix,W_matrix,training_sample[0].T)
        final_inputs,final_outputs,neuron_outputs,layered_neuron_outputs,\
            neuron_inputs = output[0],output[1],output[2],output[3],output[4]

        actual_value = vectorize_training_label(training_sample[1])
        #prediction_error: the error of the neural net's classification.
        #0 if the image is correctly classified.
        if cost_fn == "squared":
            prediction_error = 0
            for i in range(len(actual_value)):
                prediction_error += (final_outputs[i]-actual_value[i])**2
            prediction_error = 0.5*prediction_error

            W_derivative,X_derivative = differentiate_cost_fn(final_inputs,final_outputs,\
            neuron_outputs,layered_neuron_outputs,neuron_inputs,'squared',actual_value,\
            X_matrix,W_matrix,training_sample[0])

            #Taking a step in the opposite direction of the gradient:
            epoch_num = (count//epoch)+1
            learning_rate = learning_rate/epoch_num
            W_matrix = np.subtract(W_matrix,learning_rate*W_derivative)
            X_matrix = np.subtract(X_matrix,learning_rate*X_derivative)

        else:
            W_derivative,X_derivative = differentiate_cost_fn(final_inputs,final_outputs,\
            neuron_outputs,layered_neuron_outputs,neuron_inputs,'cross',actual_value,\
            X_matrix,W_matrix,training_sample[0])

            #Taking a step in the opposite direction of the gradient:
            epoch_num = (count//epoch)+1
            learning_rate = learning_rate/epoch_num
            W_matrix = np.subtract(W_matrix,learning_rate*W_derivative)
            X_matrix = np.subtract(X_matrix,learning_rate*X_derivative)

        count += 1

    if plotting:
        plt.plot(plot_x_values,plot_y_values)
        plt.show()

    if validate:
        return validate_func(validation_set,W_matrix,X_matrix)

    if test:
        predictions_list = []
        for i in range(10000):
            image = flatten((test_images[:,:,i].T))
            max_feature_value = max(image)
            max_predicted_value = 9
            image = np.append(image,[1])
            image = np.divide(image,max_feature_value)
            prediction = predict_using_weights(image,W_matrix,X_matrix)
            predictions_list.append(prediction)
        with open("cs189_hw6.csv", "w+") as int_classifier:
            writing_index = 1
            int_classifier.write("Id,Category\n")
            while writing_index < 10001:
                int_classifier.write(str(writing_index)+","+\
                    str(vector_to_number(predictions_list[writing_index-1]))+"\n")
                writing_index += 1
        return

    return (prediction_error,W_matrix,X_matrix,validation_set)

def predict_using_weights(sample,W_matrix,X_matrix):
    prediction = forward_propagation(X_matrix,W_matrix,sample.T,predicting=True)
    prediction = make_prediction_valid(prediction)
    return prediction

#Returns a random image and label from the input data.
#@param training_features: a list of images
#@param training_labels: a list of labels
def random_sample(training_features,training_labels):
    #Gets an index for the random sample:
    index = random.randint(0,len(training_features[0][0])-1)
    #Uses the index to retrieve sample and flattens it:
    random_sample = flatten(training_features[:,:,index])
    random_label = training_labels[index]
    return [random_sample,random_label,index]

#Takes a square matrix and "flattens" it.
#@param matrix: the square matrix
def flatten(matrix):
    dimension = len(matrix)
    returnVector = np.zeros(dimension**2)
    for i in range(dimension):
        for j in range(dimension):
            returnVector[i*dimension+j] = matrix[i][j]
    return returnVector

#Runs forward propagation, given the weight matrices and the random sample.
def forward_propagation(first_matrix,second_matrix,sample,predicting=False):
    output_layer_cardinality = 10
    #neuron_inputs: a vector that stores the ensembles of weights being fed
    #into each neuron. 1 is appended to it, again for bias.
    neuron_inputs = np.dot(first_matrix,sample)
    neuron_outputs = tanh(neuron_inputs)
    neuron_outputs = np.append(neuron_outputs.T,[1])
    #layered_neuron_outputs: a 10x201 array that is the neuron_outputs vector
    #layerd on top of itself ten times. This is for the matrix derivative
    #computation necessary for backpropagation.
    layered_neuron_outputs = np.array([neuron_outputs])
    for _ in range(output_layer_cardinality-1):
        layered_neuron_outputs = np.append(layered_neuron_outputs,\
            [neuron_outputs],axis=0)
    #final_inputs: a vector that stores the inputs to the final layer of the
    #neural net.
    final_inputs = np.dot(second_matrix,neuron_outputs)
    #final_outputs: final_inputs with the sigmoid applied to it.
    final_outputs = sigmoid(final_inputs)
    if not predicting:
        return (final_inputs,final_outputs,neuron_outputs,layered_neuron_outputs,\
                neuron_inputs)
    return final_outputs

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-(tanh(x)**2)

def vectorize_training_label(label):
    if label == 0: return np.array([1,0,0,0,0,0,0,0,0,0])
    elif label >= 0.11 and label <= 0.12: return np.array([0,1,0,0,0,0,0,0,0,0])
    elif label >= 0.22 and label <= 0.23: return np.array([0,0,1,0,0,0,0,0,0,0])
    elif label >= 0.33 and label <= 0.34: return np.array([0,0,0,1,0,0,0,0,0,0])
    elif label >= 0.44 and label <= 0.45: return np.array([0,0,0,0,1,0,0,0,0,0])
    elif label >= 0.55 and label <= 0.56: return np.array([0,0,0,0,0,1,0,0,0,0])
    elif label >= 0.66 and label <= 0.67: return np.array([0,0,0,0,0,0,1,0,0,0])
    elif label >= 0.77 and label <= 0.78: return np.array([0,0,0,0,0,0,0,1,0,0])
    elif label >= 0.88 and label <= 0.89: return np.array([0,0,0,0,0,0,0,0,1,0])
    else: return np.array([0,0,0,0,0,0,0,0,0,1])

def differentiate_cost_fn(final_inputs,final_outputs,neuron_outputs,\
    layered_neuron_outputs,neuron_inputs,func_name,actual_value,X_matrix,\
    W_matrix,sample):
    if func_name == 'squared':
        #For the W_derivative:
        #Dimension of neuron_outputs: (201x1)

        error_vector = (-1)*(np.subtract(actual_value,final_outputs))
        d3 = np.multiply(error_vector,sigmoid_prime(final_inputs)).reshape(10,1)
        W_derivative = np.outer(d3,neuron_outputs)
        W_matrix = np.delete(W_matrix,[200],1)
        neuron_inputs = np.expand_dims(neuron_inputs,axis=0).T
        d2 = (np.dot(W_matrix.T,d3)*tanh_prime(neuron_inputs))
        X_derivative = np.outer(d2,sample)
        return (W_derivative,X_derivative)

    else:
        #Differentiating cross-entropy loss function:
        error_vector_first = (-1)*np.divide(actual_value,final_outputs)
        error_vector_second = np.divide(np.subtract(1,actual_value),\
                np.subtract(1,final_outputs))
        error_vector_final = np.add(error_vector_first,error_vector_second)
        d3 = np.multiply(error_vector_final,sigmoid_prime(final_inputs))\
                .reshape(10,1)
        W_derivative = np.outer(d3,neuron_outputs)
        W_matrix = np.delete(W_matrix,[200],1)
        neuron_inputs = np.expand_dims(neuron_inputs,axis=0).T
        d2 = (np.dot(W_matrix.T,d3)*tanh_prime(neuron_inputs))
        X_derivative = np.outer(d2,sample)
        return (W_derivative,X_derivative)

def reshape_neuron_output(neuron_outputs,reshape_factor,three_dim=False):
    returnMatrix = np.zeros((reshape_factor,201))
    for i in range(reshape_factor):
        returnMatrix[i] = neuron_outputs
    if three_dim:
        returnMatrix = np.ones((10,201,10))
        for j in range(10):
            returnMatrix[:,:,j] = neuron_outputs
        return returnMatrix
    return returnMatrix

def make_prediction_valid(prediction):
    returnArray = np.zeros((10,), dtype=np.int)
    max_index,current_max = 0,0
    for i in range(len(prediction)):
        if current_max < prediction[i]:
            current_max = prediction[i]
            max_index = i
    returnArray[max_index] = 1
    return returnArray

def vector_to_number(vector):
    if np.array_equal(np.array([1,0,0,0,0,0,0,0,0,0]),vector): return 0
    elif np.array_equal(np.array([0,1,0,0,0,0,0,0,0,0]),vector): return 1
    elif np.array_equal(np.array([0,0,1,0,0,0,0,0,0,0]),vector): return 2
    elif np.array_equal(np.array([0,0,0,1,0,0,0,0,0,0]),vector): return 3
    elif np.array_equal(np.array([0,0,0,0,1,0,0,0,0,0]),vector): return 4
    elif np.array_equal(np.array([0,0,0,0,0,1,0,0,0,0]),vector): return 5
    elif np.array_equal(np.array([0,0,0,0,0,0,1,0,0,0]),vector): return 6
    elif np.array_equal(np.array([0,0,0,0,0,0,0,1,0,0]),vector): return 7
    elif np.array_equal(np.array([0,0,0,0,0,0,0,0,1,0]),vector): return 8
    elif np.array_equal(np.array([0,0,0,0,0,0,0,0,0,1]),vector): return 9

def shuffle(training_features,training_labels):
    data_size = len(training_labels)
    returnData,returnLabel = [],np.zeros(data_size)
    random_numbers = np.arange(data_size)
    np.random.shuffle(random_numbers)
    index = 0
    for num in random_numbers:
         returnData.append(flatten(training_features[:,:,num]))
         returnLabel[index] = training_labels[num]
         index += 1
    return [returnData,returnLabel]

def make_three_dim_weights(matrix):
    returnMatrix = np.zeros((200,785,10))
    for i in range(200):
        assign_vector = matrix[:,i]
        for j in range(785):
            returnMatrix[i][j] = assign_vector
    return returnMatrix

def make_error_scalar_matrix(predicted,actual):
    returnMatrix = np.zeros((10,201))
    for i in range(10):
        assign_vector = -1*(actual[i]-predicted[i])*np.ones(201)
        returnMatrix[i,:] = assign_vector
    return returnMatrix

def reshape_data_sample(sample):
    returnMatrix = np.zeros((785,785))
    for i in range(785):
        returnMatrix[i] = sample
    return returnMatrix

def validate_func(validation_set,W_matrix,X_matrix):
    error_count = 0
    for lst in validation_set:
        prediction = predict_using_weights(lst[0],W_matrix,X_matrix)
        if not np.array_equal(lst[1],prediction):
            error_count += 1
    validation_error = error_count/len(validation_set)
    print("Validation error is: " + str(validation_error))
    return validation_error
