# Restricted Boltzmann Machine
This project builds a RMB. 

## How to use the model
### Install required packages 
tensorflow, numpy, time, os, matplotlib
```
sh requirements.sh
```


### How to use
Example is in `main.py`. In a python file <br />  
Extract MNIST data 
```
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
```
Create a Restricted Boltzmann Machine with MNIST training data 
```
rbm = RBM(num_visible=784, num_hidden=10,
      train_data=mnist.train.images, epochs=20,
      learning_rate=0.00001)
```
Train the Restricted Boltzmann Machine with the training data
```
rbm.train()
```
### Results 
Number|GIF  
-----------:|:----------------------------:
0|![alt text](./pictures/0.gif)
1|![alt text](./pictures/1.gif)
2|![alt text](./pictures/2.gif)
3|![alt text](./pictures/3.gif)
4|![alt text](./pictures/4.gif)
5|![alt text](./pictures/5.gif)
6|![alt text](./pictures/6.gif)
7|![alt text](./pictures/7.gif)
8|![alt text](./pictures/8.gif)
9|![alt text](./pictures/9.gif)

