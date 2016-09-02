from sklearn.cross_validation import train_test_split


from gensim.models.doc2vec import Doc2Vec
import networkutils as net

from random import shuffle

import Evaluation

class TriDNR:
    """
        Tri-Party Deep Network Representation, IJCAI-2016

    Read the data from a `directory` which contains text, label, structure information, and initialize the TriDNR from
    Doc2Vec and DeepWalk Models, then iteratively update the model with text, label, and structure information

    The `directory` read the data from the directory 'dir', the directory should contain three files:
        docs.txt -- text document for each node, one line for one node
        labels.txt -- class label for each node, one line for one node
        adjedges.txt -- edge list of each node, one line for one node.


    `train_size`: Percentage of training data in range 0.0-1.0, if train_size==0, it becomes pure unsupervised network
    representation

    `text_weight`: weights for the text inforamtion 0.0-1.0

    `size` is the dimensionality of the feature vectors.


    `dm` defines doc2vec the training algorithm.  (`dm=1`), 'distributed memory' (PV-DM) is used.
    Otherwise, `distributed bag of words` (PV-DBOW) is employed.

    `min_count`: minimum number of counts for words


     """
    def __init__(self, directory=None, train_size=0.3, textweight=0.8, size=300, seed=1, workers=1, passes=10, dm=0, min_count=3):

        # Read the data
        alldocs, docindex, classlabels = net.readNetworkData(directory)
        print('%d documents, %d classes, training ratio=%f' % (len(alldocs), len(classlabels), train_size))
        print('%d classes' % len(classlabels))

        #Initilize Doc2Vec
        if train_size  > 0: #label information is available for learning
            print('Adding Label Information')
            train, test = train_test_split(alldocs, train_size=train_size, random_state=seed)

            """
                Add supervised information to training data, use label information for learning
                Specifically, the doc2vec algorithm used the tags information as document IDs,
                and learn a vector representation for each tag (ID). We add the class label into the tags,
                so each class label will acts as a ID and is used to learn the latent representation
            """
            alldata = train[:]
            for x in alldata:
                x.tags.append('Label_'+x.labels)
            alldata.extend(test)
        else: # no label information is available, pure unsupervised learning
            alldata = alldocs[:]


        d2v = net.trainDoc2Vec(alldata, workers=workers, size=size, dm=dm, passes=passes, min_count=min_count)

        raw_walks, netwalks = net.getdeepwalks(directory, number_walks=20, walk_length=8)
        w2v = net.trainWord2Vec(raw_walks, buildvoc=1, passes=passes, size=size, workers=workers)

        if train_size > 0: #Print out the initial results
            print('Initialize Doc2Vec Model With Supervised Information...')
            Evaluation.evaluationEmbedModelFromTrainTest(d2v, train, test, classifierStr='SVM')
            print('Initialize Deep Walk Model')
            Evaluation.evaluationEmbedModelFromTrainTest(w2v, train, test, classifierStr='SVM')

        self.d2v = d2v
        self.w2v = w2v

        self.train(d2v, w2v, directory, alldata, passes=passes, weight=textweight)

        if textweight > 0.5:
            self.model = d2v
        else:
            self.model = w2v


    def setWeights(self, orignialModel, destModel, weight=1):

        if isinstance(orignialModel, Doc2Vec):
            print('Copy Weights from Doc2Vec to Word2Vec')
           # destModel.reset_weights()

            doctags = orignialModel.docvecs.doctags
            keys = destModel.vocab.keys()
            for key in keys:
                if not doctags.__contains__(key):
                    continue

                index = doctags[key].index # Doc2Vec index
                id = destModel.vocab[key].index # Word2Vec index
                destModel.syn0[id] = (1-weight) * destModel.syn0[id] + weight * orignialModel.docvecs.doctag_syn0[index]

                destModel.syn0_lockf[id] = orignialModel.docvecs.doctag_syn0_lockf[index]
        else: # orignialModel is a word2vec instance only
            print('Copy Weights from Word2Vec to Doc2Vec')
            assert isinstance(destModel, Doc2Vec)
            doctags = destModel.docvecs.doctags
            keys = orignialModel.vocab.keys()
            for key in keys:
                if not doctags.__contains__(key):
                    continue
                index = doctags[key].index # Doc2Vec index
                id = orignialModel.vocab[key].index # Word2Vec index
                destModel.docvecs.doctag_syn0[index] = (1-weight) * destModel.docvecs.doctag_syn0[index] + weight * orignialModel.syn0[id]
                destModel.docvecs.doctag_syn0_lockf[index] = orignialModel.syn0_lockf[id]


    def train(self, d2v, w2v, directory, alldata, passes=10, weight=0.9):

        raw_walks, walks = net.getdeepwalks(directory, number_walks=20, walk_length=10)
        for i in xrange(passes):
            print('Iterative Runing %d' % i)
            self.setWeights(d2v, w2v, weight=weight)

            #Train Word2Vec

            shuffle(raw_walks)
            print("Update W2V...")
            w2v.train(raw_walks)
            self.setWeights(w2v, d2v, weight=(1-weight))

            print("Update D2V...")
            shuffle(alldata)  # shuffling gets best results
            d2v.train(alldata)
