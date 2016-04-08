import interface as bbox
import theano
import numpy as np
import theano.tensor as T
import lasagne
import theano.tensor.nnet as nnet

#input will be x, output is y
x = T.dvector()
y = T.dscalar()
#layer-takes x and weight matrix, multiplies them
# and then runs through sigmoid function
def layer(x, w):
    b = np.array([1], dtype=theano.config.floatX)
    new_x = T.concatenate([x, b])
    m = T.dot(w.T, new_x) #theta1: 3x3 * x: 3x1 = 3x1 ;;; theta2: 1x4 * 4x1
    h = nnet.sigmoid(m)
    return h
#gradient descent
def grad_desc(cost, theta):
    alpha = 0.1 #learning rate
    return theta - (alpha * T.grad(cost, wrt=theta))
#A shared variable is what we use for things we want to 
#give a definite value but we also want to update.

#here we are defining weight matrices, random initialization

theta1 = theano.shared(np.array(np.random.rand(3,5), dtype=theano.config.floatX)) # randomly initialize
theta2 = theano.shared(np.array(np.random.rand(6,5), dtype=theano.config.floatX))

hid1 = layer(x, theta1) #hidden layer

out1 = layer(hid1, theta2) #T.sum(layer(hid1, theta2)) #output layer

error = T.sum((out1 - y)**2) #error

cost = theano.function(inputs=[x, y], outputs=error, updates=[
        (theta1, grad_desc(error, theta1)),
        (theta2, grad_desc(error, theta2))]) 
#output layer expression
run_forward = theano.function(inputs=[x], outputs=out1)

inputs = np.array([[0,1],[1,0],[1,1],[0,0]]).reshape(4,2) #training data X
exp_y = np.array([1, 1, 0, 0]) #training data Y
cur_cost = 0
for i in range(10000):
    for k in range(len(inputs)):
        cur_cost = cost(inputs[k], exp_y[k]) #call our Theano-compiled cost function, it will auto update weights
    if i % 500 == 0: #only print the cost every 500 iterations
        print('Cost: %s' % (cur_cost,))
print(run_forward([0,1]))
print(run_forward([1,1]))
print(run_forward([1,0]))
print(run_forward([0,0]))
