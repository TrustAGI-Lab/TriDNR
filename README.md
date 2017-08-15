# TriDNR
Tri-Party Deep Network Representation

The codes implement the TriDNR algorithm, which learns an continuous representation for each node in a network. TriDNR uses information from three perspectives, including node structure, node content, and node labels (if available), to jointly learn optimal node representation


The code is developed in Python, based on the package gensim, and DeepWalk. All required packages are defined in requirements.txt. To install all requirement, simply use the following commands:

	pip install -r requirements.txt

A demo is provide in 'demo.py', which runs and compares several algorithms 

About the datasets:
There are two networked datasets in the paper, i.e., DBLP and Citeseer-M10.
Each dataset containts 3 files:
	
	1. docs.txt : title information of each node in a network, each line represents a node (paper). The first item in each line is the node ID

	2. adjedges.txt : neighbor nodes of each node in a network. The first item in each line is the node ID, and the rest items are nodes that has a link to the first node. Node that if only one item in a line, it means that the node has no links to other nodes

	3.labels: class labels of a node. Each line represents a node id and its class label

The first item of each line across three files are matched. 


Note:
1. On some dataset (M10), some neighbor nodes do not appear in the files of docs.txt or labels.txt. 
2. For the fairness of comparison, I would sugget shuffling the dataset before training different methods. 
