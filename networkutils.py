from collections import namedtuple
from gensim.models.doc2vec import Doc2Vec, Word2Vec
from random import shuffle
from deepwalk import graph
import gensim
import random

import gensim.utils as ut


NetworkSentence=namedtuple('NetworkSentence', 'words tags labels index')
Result = namedtuple('Result', 'alg trainsize acc macro_f1 micro_f1')
AlgResult = namedtuple('AlgResult', 'alg trainsize numfeature mean std')



def readNetworkData(dir, stemmer=0): #dir, directory of network dataset
    allindex={}
    alldocs = []
    labelset = set()
    with open(dir+'/docs.txt') as f1, open(dir + '/labels.txt') as f2:

        for l1 in f1:
            #tokens = ut.to_unicode(l1.lower()).split()
            if stemmer == 1:
                l1 = gensim.parsing.stem_text(l1)
            else:
                l1 = l1.lower()
            tokens = ut.to_unicode(l1).split()

            words = tokens[1:]
            tags = [tokens[0]] # ID of each document, for doc2vec model
            index = len(alldocs)
            allindex[tokens[0]] = index # A mapping from documentID to index, start from 0

            l2 = f2.readline()
            tokens2 = gensim.utils.to_unicode(l2).split()
            labels = tokens2[1] #class label
            labelset.add(labels)
            alldocs.append(NetworkSentence(words, tags, labels, index))

    return alldocs, allindex, list(labelset)

def coraEdgeFileToAdjfile(edgefile, adjfile, nodoc):
    edgeadj = {str(n): list() for n in xrange(nodoc)}
    with open(edgefile, 'r') as f:
        for l in f:
            tokens = l.split()
            edgeadj[tokens[0]].append(tokens[1])

    wf = open(adjfile, 'w')
    for n in xrange(nodoc):
        edgestr = ' '.join(map(str, edgeadj[str(n)]))
        wf.write(str(n) + ' ' +edgestr +'\n')
    wf.close()

def cora10groupdataset():
    groupindex = {}
    groupmap = {}
    with open('data2/Cora/CoraHierarchyTree.txt', 'r') as f:
        for l in f:
            tokens = l.split('\t')
            if(len(tokens) <= 2):
                continue
            elif(len(tokens) == 3):
                currentindex = len(groupindex)
                groupindex[tokens[1]] = currentindex

            elif(len(tokens) == 5):
                #print l
                groupmap[tokens[3]] = currentindex
            elif(len(tokens)==6):
                #print tokens[3]
                groupmap[tokens[4]] = currentindex
            else:
                pass

    # All class 0 as a separate class
    groupmap['0'] = len(groupindex) + 1

    print('number of classes: %d ' % len(groupmap))

    wf = open('data2/Cora/labels.txt', 'w')
    with open('data2/Cora/paper_label.txt') as f1:
        for l in f1:
            tokens = l.split()
            wf.write(tokens[0] + ' ' + str(groupmap[tokens[1]]) + '\n')

    wf.close()


def getdeepwalks(dir, number_walks=50, walk_length=10, seed=1):
    G = graph.load_adjacencylist(dir+'/adjedges.txt')

    print("Number of nodes: {}".format(len(G.nodes())))
    num_walks = len(G.nodes()) * number_walks
    print("Number of walks: {}".format(num_walks))

    print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=number_walks, path_length=walk_length, alpha=0, rand=random.Random(seed))
    networksentence = []
    raw_walks = []
    for i, x in enumerate(walks):
        sentence = [gensim.utils.to_unicode(str(t)) for t in x]

        s = NetworkSentence(sentence, [sentence[0]], None, 0) # label information is not used by random walk
        networksentence.append(s)
        raw_walks.append(sentence)

    return raw_walks, networksentence

def trainDoc2Vec(doc_list=None, buildvoc=1, passes=20, dm=0,
                 size=100, dm_mean=0, window=5, hs=1, negative=5, min_count=1, workers=4):
    model = Doc2Vec(dm=dm, size=size, dm_mean=dm_mean, window=window,
                    hs=hs, negative=negative, min_count=min_count, workers=workers) #PV-DBOW
    if buildvoc == 1:
        print('Building Vocabulary')
        model.build_vocab(doc_list)  # build vocabulate with words + nodeID

    for epoch in range(passes):
        print('Iteration %d ....' % epoch)
        shuffle(doc_list)  # shuffling gets best results

        model.train(doc_list)

    return model


def trainWord2Vec(doc_list=None, buildvoc=1, passes=20, sg=1, size=100,
                  dm_mean=0, window=5, hs=1, negative=5, min_count=1, workers=4):
    model = Word2Vec(size=size,  sg=sg,  window=window,
                     hs=hs, negative=negative, min_count=min_count, workers=workers)

    if buildvoc == 1:
        print('Building Vocabulary')
        model.build_vocab(doc_list)  # build vocabulate with words + nodeID

    for epoch in range(passes):
        print('Iteration %d ....' % epoch)
        shuffle(doc_list)  # shuffling gets best results

        model.train(doc_list)

    return model

def toMatFile(directory, tfidf=1): #convert the file to a matlab file, using TF-IDF or Binary format
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_selection import SelectKBest, chi2
    alldocs, allsentence, classlabels = readNetworkData(directory)

    corpus = [' '.join(doc.words) for doc in alldocs]

    vectorizer = TfidfVectorizer(min_df=3)

    ## Feature Vector
    new_vec = vectorizer.fit_transform(corpus)
    print("n_samples: %d, n_features: %d" % new_vec.shape)

    ## Label
    y = [doc.labels for doc in alldocs]

    ## Vector
