## Python Code to generate random numbers
```
import numpy 
np.random.seed(0)
wh=np.random.uniform(0,1,12).reshape(4,3)

np.random.seed(2)
bh=np.random.uniform(0,1,3)
array([ 0.4359949 ,  0.02592623,  0.54966248])

np.random.seed(3)
wout=np.random.uniform(0,1,3)
array([ 0.4359949 ,  0.02592623,  0.54966248])

np.random.seed(4)
bout=np.random.uniform(0,1,1)
```

## Wh
![deeplearning/kernel.PNG at master · shanky221341/deeplearning · GitHub](https://github.com/shanky221341/deeplearning/raw/master/kernel.PNG)

## bh
array([ 0.4359949 ,  0.02592623,  0.54966248])

## wout
array([ 0.5507979 ,  0.70814782,  0.29090474])

## bout
array([ 0.96702984])

### Backpropagation steps

#### Step0: Read input and output

![deeplearning/step1.PNG at master · shanky221341/deeplearning · GitHub](https://github.com/shanky221341/deeplearning/raw/master/step1.PNG =1000x60)

#### Step1: Initialize weights and biases with random values using python code above

![deeplearning/step1.PNG at master · shanky221341/deeplearning · GitHub](https://github.com/shanky221341/deeplearning/raw/master/step2.PNG =1000x60)

#### Step2: Calculate hidden layer input:
hidden_layer_input = matrix_dot_product(X,wh) + bh

![deeplearning/step1.PNG at master · shanky221341/deeplearning · GitHub](https://github.com/shanky221341/deeplearning/raw/master/step3.PNG =1000x60)

#### Step3:Perform non-linear transformation on hidden linear input


hiddenlayer_activations = sigmoid(hidden_layer_input)

![deeplearning/step1.PNG at master · shanky221341/deeplearning · GitHub](https://github.com/shanky221341/deeplearning/raw/master/step4.PNG =1000x60)

#### Step4: Perform linear and non-linear transformation of hidden layer activation at output layer

output_layer_input = matrix_dot_product (hiddenlayer_activations * wout ) + bout output = sigmoid(output_layer_input)

![deeplearning/step1.PNG at master · shanky221341/deeplearning · GitHub](https://github.com/shanky221341/deeplearning/raw/master/step5.PNG =1000x60)

#### Step5 Calculate gradient of Error(E) at output layer

E = y-output

![deeplearning/step1.PNG at master · shanky221341/deeplearning · GitHub](https://github.com/shanky221341/deeplearning/raw/master/step6.PNG =1000x60)

#### Step6  Compute slope at output and hidden layer
Slope_output_layer= derivatives_sigmoid(output) =x*(1-x)

Slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations)=x*(1-x)

![deeplearning/step1.PNG at master · shanky221341/deeplearning · GitHub](https://github.com/shanky221341/deeplearning/raw/master/step7.PNG )

#### Step7 Compute delta at output layer
learning rate=0.1
d_output = E * slope_output_layer*learning rate

![deeplearning/step1.PNG at master · shanky221341/deeplearning · GitHub](https://github.com/shanky221341/deeplearning/raw/master/step8.PNG)

#### Step8 Calculate Error at hidden layer
Error_at_hidden_layer = matrix_dot_product(d_output, wout.Transpose)

![deeplearning/step1.PNG at master · shanky221341/deeplearning · GitHub](https://github.com/shanky221341/deeplearning/raw/master/step9.PNG)


#### Step9 Compute delta at hidden layer

d_hiddenlayer = Error_at_hidden_layer * slope_hidden_layer

![deeplearning/step1.PNG at master · shanky221341/deeplearning · GitHub](https://github.com/shanky221341/deeplearning/raw/master/step10.PNG )

#### Step10 Update weight at both output and hidden layer 

wout = wout + matrix_dot_product (hiddenlayer_activations.Transpose, d_output) * learning_rate

wh = wh+ matrix_dot_product (X.Transpose,d_hiddenlayer) * learning_rate

![deeplearning/step1.PNG at master · shanky221341/deeplearning · GitHub](https://github.com/shanky221341/deeplearning/raw/master/step11.PNG)

#### Step11 Update biases at both output and hidden layer

bh = bh + sum(d_hiddenlayer, axis=0) * learning_rate

bout = bout + sum(d_output, axis=0)*learning_rate

![deeplearning/step1.PNG at master · shanky221341/deeplearning · GitHub](https://github.com/shanky221341/deeplearning/raw/master/step12.PNG)