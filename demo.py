"""
A Demo Comparing Several Network Representation Algorithms:

Doc2Vec: Paragraph Vector model which only use text information
DeepWalk: DeepWalk model which only use structure information
Doc2Vec+DeepWalk: Simple combination of Doc2Vec and DeepWalk Model

TriDNR: Tri-Party Deep Network Representation Model, published in IJCAI-2016

"""

import networkutils as net
from sklearn.cross_validation import train_test_split
import numpy as np

import Evaluation

numFea = 100
cores = 4
train_size = 0.2  #Percentage of training samples
random_state = 2
dm = 0
passes = 20

directory = 'data/M10' #data directory
alldocs, allsentence, classlabels = net.readNetworkData(directory)
print('%d documents' % len(alldocs))
print('%d classes' % len(classlabels))
doc_list = alldocs[:]  # for reshuffling per pass


train, test = train_test_split(doc_list, train_size=train_size, random_state=random_state)


### Baseline 1, Doc2Vec Model (PV-DM)
print("##################")
print("Baseline 1, Doc2Vec Model dm=%d" % dm)
doc2vec_model = net.trainDoc2Vec(doc_list, workers=cores, size=numFea, dm=dm, passes=passes, min_count=3)

print('Classification Performance on Doc2Vec Model')
doc2vec_acc, doc2vec_macro_f1, doc_2vec_micro_f1 = \
     Evaluation.evaluationEmbedModelFromTrainTest(doc2vec_model, train, test, classifierStr='SVM')

print("##################")


#### Baseline 2, Deep Walk Model
print("##################")
print("Baseline 2, Deep Walk Model")
raw_walks, netwalks = net.getdeepwalks(directory, number_walks=20, walk_length=8)
deepwalk_model = net.trainWord2Vec(raw_walks, buildvoc=1, sg=1, passes=passes, size=numFea, workers=cores)
print('Classification Performance on DeepWalk Model')
doc2vec_acc, doc2vec_macro_f1, doc_2vec_micro_f1 = \
    Evaluation.evaluationEmbedModelFromTrainTest(deepwalk_model, train, test, classifierStr='SVM')

print("##################")




### Baseline 3, D2V+DW
print("##################")
print("Baseline 3, Simple Combination of DeepWalk + Doc2Vec")

d2v_train_vecs = [doc2vec_model.docvecs[doc.tags[0]] for doc in train]
d2v_test_vecs = [doc2vec_model.docvecs[doc.tags[0]] for doc in test]

dw_train_vecs = [deepwalk_model[doc.tags[0]] for doc in train]
dw_test_vecs = [deepwalk_model[doc.tags[0]] for doc in test]

train_y = [doc.labels for doc in train]
test_y = [doc.labels for doc in test]


#concanate two vector
train_vecs = [np.append(l, dw_train_vecs[i]) for i, l in enumerate(d2v_train_vecs)]
test_vecs = [np.append(l, dw_test_vecs[i]) for i, l in enumerate(d2v_test_vecs)]


print('train y: , test y: ', len(train_y), len(test_y))
print('Classification Performance on Doc2Vec + DeepWalk')

acc, macro_f1, micro_f1 = Evaluation.evaluation(train_vecs, test_vecs, train_y, test_y, 'SVM')
print("##################")

### Our method, TriDNR
#train_size=0.3
from tridnr import TriDNR
tridnr_model = TriDNR(directory, size=numFea, dm=0, textweight=.8, train_size=train_size, seed=random_state, passes=10)
Evaluation.evaluationEmbedModelFromTrainTest(tridnr_model.model, train, test, classifierStr='SVM')
