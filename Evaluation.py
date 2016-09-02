from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.cross_validation import train_test_split
from gensim.models.doc2vec import Doc2Vec
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing




def evaluation(train_vec, test_vec, train_y, test_y, classifierStr='SVM', normalize=0):

    if classifierStr == 'KNN':
        print('Training NN classifier...')
        classifier = KNeighborsClassifier(n_neighbors=1)
    else:
        print('Training SVM classifier...')

        classifier = LinearSVC()

    if(normalize == 1):
        print('Normalize data')
        allvec = list(train_vec)
        allvec.extend(test_vec)
        allvec_normalized = preprocessing.normalize(allvec, norm='l2', axis=1)
        train_vec = allvec_normalized[0:len(train_y)]
        test_vec = allvec_normalized[len(train_y):]


    classifier.fit(train_vec, train_y)
    y_pred = classifier.predict(test_vec)
    cm = confusion_matrix(test_y, y_pred)

    print(cm)
    acc = accuracy_score(test_y, y_pred)
    print(acc)

    #macro_f1 = f1_score(test_y, y_pred,pos_label=None, average='macro')
    #micro_f1 = f1_score(test_y, y_pred,pos_label=None, average='micro')

    macro_f1 = f1_score(test_y, y_pred,pos_label=None, average='macro')
    micro_f1 = f1_score(test_y, y_pred,pos_label=None, average='micro')

    per = len(train_y) * 1.0 /(len(test_y)+len(train_y))
    print('Classification method:'+classifierStr+'(train, test, Training_Percent): (%d, %d, %f)' % (len(train_y),len(test_y), per ))
    print('Classification Accuracy=%f, macro_f1=%f, micro_f1=%f' % (acc, macro_f1, micro_f1))
    #print(metrics.classification_report(test_y, y_pred))

    return acc, macro_f1, micro_f1

def evaluationEmbedModel(model, alldocs, train_size=0.2, random_state=1, classifierStr='SVM', normalize=0):
    train, test = train_test_split(alldocs, train_size=train_size, random_state=random_state)

    return evaluationEmbedModelFromTrainTest(model, train, test, classifierStr, normalize)


def evaluationEmbedModelFromTrainTest(model, train, test, classifierStr='SVM', normalize=0):

    if isinstance(model, Doc2Vec):
        train_vecs = [model.docvecs[doc.tags[0]] for doc in train]
        test_vecs = [model.docvecs[doc.tags[0]] for doc in test]
    else: #Word2Vec model
        train_vecs = [model[doc.tags[0]] for doc in train]
        test_vecs = [model[doc.tags[0]] for doc in test]

    train_y = [doc.labels for doc in train]
    test_y = [doc.labels for doc in test]

    print('train y: , test y: ', len(train_y), len(test_y))

    acc, macro_f1, micro_f1 = evaluation(train_vecs, test_vecs, train_y, test_y, classifierStr, normalize)

    return acc, macro_f1, micro_f1

def evaluateConcanateModel(doc2vec_model, model2, train, test, normalize=0):
    d2v_train_vecs = [doc2vec_model.docvecs[doc.tags[0]] for doc in train]
    d2v_test_vecs = [doc2vec_model.docvecs[doc.tags[0]] for doc in test]

    if isinstance(model2, Doc2Vec):
        dw_train_vecs = [model2.docvecs[doc.tags[0]] for doc in train]
        dw_test_vecs = [model2.docvecs[doc.tags[0]] for doc in test]
    else:
        dw_train_vecs = [model2[doc.tags[0]] for doc in train]
        dw_test_vecs = [model2[doc.tags[0]] for doc in test]

    train_y = [doc.labels for doc in train]
    test_y = [doc.labels for doc in test]

    import numpy as np
    #concanate two vector
    train_vecs = [np.append(l, dw_train_vecs[i]) for i, l in enumerate(d2v_train_vecs)]
    test_vecs = [np.append(l, dw_test_vecs[i]) for i, l in enumerate(d2v_test_vecs)]


    print('train y: , test y: ', len(train_y), len(test_y))
    print('Classification Performance on Doc2Vec + DeepWalk')

    con_acc, con_macro_f1, con_micro_f1 = evaluation(train_vecs, test_vecs, train_y, test_y, 'SVM', normalize)
    return con_acc, con_macro_f1, con_micro_f1
