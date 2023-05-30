# Neural Network Dependency Parser
This project trains a feed-forward neural network to predict the transitions of an arc-standard dependency parser. The Input to this network will be a representation of the current state (including words on the stack and buffer). The Output will be a transition (shift, left_arc, right_arc), together with a dependency relation label. It uses TensorFlow and Keras package to construct the neural net. 
The project include the following parts:
- extracting Input/Output matrices for training
- designing and training the network
- greedy parsing algorithm

## Usage
Run the Jupyter Notebook run.ipynb in Google Colab

## Data
The data come from a standard split of the WSJ part of the Penn Treebank. Data structure to represent, read, and write dependency trees is in the CoNLL-X format. 
