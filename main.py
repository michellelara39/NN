
import numpy as np
input_num = 13
hid_num = 120
out_num = 3
def input_Read():

    file = open(r'C:\Users\Michelle Lara\PycharmProjects\Project_DL_3\wine.data.txt')
    #file = open(r'C:\Users\Michelle Lara\PycharmProjects\Project_DL_3\Inputs.txt')
    data = []
    for line in file.readlines():
        data.append([float(x) for x in line.replace('\n', '').split(',')])
    file.close()
    input_data = np.array(data[:])[:, 1*(out_num):]
    target_out = np.array(data[:])[:, :1*(out_num)]
    return input_data, target_out

def resultWbias(input, hid, bias):
    input[0] *= bias[0]
    i = 1
    #print('INPUT',input)
    for i in range(hid_num):
        hid = np.insert(hid, 0, bias[i])
    #hid = np.insert(hid, 0, bias[2])
    return input, hid

def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result

def dotAndSigCompute(input, weight):
   
    result = np.dot(input, weight)
    out = sigmoid(result) 
    return out

def feedForwardProcess(input, hidweight, outweight):
    #print('hid weight at feed forward', hidweight)
    #print('out weight at feed forward', outweight)
    #print('input', input)
    hid_result = dotAndSigCompute(input, hidweight)
    result = np.array(hid_result)
    result=biasAdder(result)
    outresult = dotAndSigCompute(result, outweight)
    out = outresult.flatten()

    return out, hid_result

def flat(output, target, out_weight, hid_result, hid_weight, input):
    output = output.flatten()
    target = target.flatten()
    out_weight = out_weight.T
    hid_result = hid_result.flatten()
    #hid_weight = hid_weight.flatten()
    input = input.flatten()
    return output, target, out_weight, hid_result, hid_weight, input

def report(target, out):
    tolerance = 0.2
    #print('target', target)
    #print('out', out)
    #out = np.transpose(out) 
    hits = np.abs(target - out) <= tolerance
    #print('hits' ,hits)
    accuracy = (np.sum(hits) / len(target)) * 100
    return accuracy, hits

def check(target, out):

    difference = np.abs(target - out)
    difference = difference * 100
    #print('difference', difference)
    return difference

def biasAdder(function):

    if function.ndim == 1:
        # For 1D array, insert 1 at the beginning
        function = np.insert(function, 0,1)
    elif function.ndim == 2:
        # For 2D array, insert a column of ones at the beginning
        function = np.insert(function, [0], 1, axis=1)
    return function

def derivative(output):
    deriv = output * (1.0 - output)
    return deriv

def error(output, target):
    err = (target - output) * derivative(output)
    return err

def organize(output, target, out_weight, hid_result, hid_weight, input,bias):
    output = output.reshape(1, -1)
    target = target.reshape(1, -1)
    out_weight = out_weight[1:]
    hid_result = hid_result.reshape(1, -1)
    hid_weight = hid_weight[1:]
    input = input[1:]
    input = input.reshape(1, -1)
    return output, target, out_weight, hid_result, hid_weight, input,bias

def show(output, target, out_weight,bias, hid_result, hid_weight, input):
    print('output',output)
    target = np.array(target)
    print('target',target)
    print('out weight', out_weight)
    print('bias', bias)
    print('hid result', hid_result)
    print('hid weight', hid_weight)
    print('input', input)

def backProp(output, target, out_weight,bias, hid_result, hid_weight, input, learn):
    #print('\n')
    
    output, target, out_weight, hid_result, hid_weight, input,bias =organize(output, target, out_weight, hid_result, hid_weight, input,bias)
    #show(output, target, out_weight,bias, hid_result, hid_weight, input)
    # δ_out = derivative(output) * (target - output)
    output_delta = error(output, target) 

    # Hidden layer error
    hidden_error = np.dot( output_delta,out_weight.T)
    hidden_delta = hidden_error * derivative(hid_result)
    #hidden_delta = hidden_delta.reshape(-1, 1)

    # Update weights and biases
    out_weight += np.dot( hid_result.T, output_delta) * learn
    out_weight = out_weight.T
    bias_out = bias[hid_num: out_num + hid_num] + output_delta * learn
    hid_weight += np.dot(input.T, hidden_delta) * learn
    bias_hidden = bias[0:hid_num] + hidden_delta * learn

    # Place bias into weights  
    out_weight = np.concatenate((bias_out.T, out_weight), axis=1)
    hid_weight = np.concatenate((bias_hidden, hid_weight), axis=0)
    bias = np.concatenate((bias_hidden.flatten(), bias_out.flatten())) 

    output, target, out_weight, hid_result, hid_weight, input = flat(output, target, out_weight, hid_result, hid_weight, input)
    #show(output, target, out_weight,bias, hid_result, hid_weight, input)
    #print('Last of BP\n')

    return out_weight, hid_weight

def main():
    # I am just creating the arrays from the .txt files
    input_data, target_out = input_Read()
    #hid_weight, out_weight,bias = weights()
    #print(hid_weight)
    #print(out_weight)
    
    input_size = input_num
    hidden_size = hid_num
    output_size = out_num

    # Initialize weights
    hid_weight = np.random.randn(input_size +1, hidden_size)
    out_weight = np.random.randn(hidden_size +1, output_size)
    #print(hid_weight)
    #print(out_weight)

    # Initialize the biases
    bias = np.concatenate((hid_weight[0][:], out_weight[0][:])) 

    # the following line will add the bias neuron
    input_data=biasAdder(input_data)
    epochs = 1400
    learn = 0.7
    out_results = []

    for _ in range(epochs):
        out_results = []
        for i in range(len(input_data)):
            input = input_data[i:i+1]
            input = input.flatten()
            target = target_out[i]
            out_result , hid_result = feedForwardProcess(input, hid_weight, out_weight)
            out_results.append(out_result.flatten())
            #difference = check(target_out[i], out_results[i])
            #if difference > 20:
            out_weight, hid_weight = backProp(out_result, target, out_weight,bias, hid_result, hid_weight, input, learn)

    #accuracy, hits = report(target_out, out_results)       
    #print('result after epochs:', out_results) 
    result = np.array(out_results)
    tar = np.array(target_out)
    result_1 = result[:, 0]
    result_2 = result[:, 1]
    result_3 = result[:, 2]
    tar_1 = tar[:, 0]
    tar_2 = tar[:, 1]
    tar_3 = tar[:, 2]
    accuracy1, _ = report(tar_1, result_1)
    accuracy2, _ = report(tar_2, result_2)
    accuracy3, _ = report(tar_3, result_3)
    print("Accuracy of out 1:", accuracy1)
    print("Accuracy of out 2:", accuracy2)
    print("Accuracy of out 3:", accuracy3)
    _, hits = report(target_out, out_results)
    print(hits)
    #print('out weight\n',out_weight)     
    #print('hid weight\n',hid_weight.T) 
    


if __name__ == "__main__":
    main()

