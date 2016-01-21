import nltk
import csv
import itertools
import ipdb
import numpy as np
import pickle

vocabulary_size = 8000
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"

with open('data/reddit-comments-2015-08.csv','rb') as f:

    reader = csv.reader(f, skipinitialspace=True)
    reader.next()
    sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader]) 
    sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    
    print "%d sentences" % len(sentences)

    tokenized_sentences = [nltk.word_tokenize(sen) for sen in sentences]
    
    # word frequency
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    
    vocab = word_freq.most_common(vocabulary_size-1)
    idx2word = [x[0] for x in vocab]
    idx2word.append(unknown_token)
    
    word2idx = dict([(w,i) for i,w in enumerate(idx2word)])

    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in idx2word else unknown_token for w in sent]

    X_train = np.asarray([[word2idx[w] for w in sent[:-1]] for sent in tokenized_sentences])
    pickle.dump(X_train, open('X_train.pk','w'))
    Y_train = np.asarray([[word2idx[w] for w in sent[1:]] for sent in tokenized_sentences]) 
    pickle.dump(Y_train, open('Y_train.pk','w'))

