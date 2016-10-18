
# coding: utf-8

# In[ ]:

## Making a dictionary from a input file of text
## lowercase unique entries

import numpy as np
import theano
import theano.tensor as T
import nltk
import string
from collections import defaultdict

rng = np.random


# 
# 
# ## Helper functions for text analysis

# In[ ]:

#------------------------------------------
# A useful function in testing
def linecount(tfile):
    with open(tfile) as f:
        i=0
        for line in f:
            i+=1
    return i 

# The words in simplelist are too common to 
# have syntatic meaning
simplelist = set('for a of the this that these those and to in'.split())

#------------------------------------------
# makes a cutoff dictionary from a textfile
# we don't need a full wordlist, we will pioche words 
# from text as we process the data.
def textfile_worddict(tfile, lines):
    lc=0
    wordcounts=defaultdict(int)
    with open(tfile, 'r') as f:
        for line in f:
            lc+=1
            line=(line.lower()).translate(None, string.punctuation)
            for word in nltk.word_tokenize(line):
                if word not in simplelist:
                    wordcounts[word]+=1
            if lc==lines:
                break
    return wordcounts

#------------------------------------------
def dict_cutoff(ddict,cutoff):
    wccutoff=defaultdict(int)
    i=0
    for key in ddict:
        if ddict[key] > cutoff:
            wccutoff[key]=i
            i+=1
    return wccutoff

#------------------------------------------
# makes a unit vector in numpy
def unit_vector(index,length):
    unitvec=np.eye(1,length,index)
    return unitvec


#------------------------------------------
# helper function to take a bag of words and returns a numpy array of size (0,dlen)
# which is the sum of the representative vectors for each word
def container_sum(contnr,dlen):
    contsum=np.zeros(dlen)
    for i in xrange(len(contnr)):
        contsum=np.add(contsum,unit_vector(contnr[i],dlen))
    contav=(1/float(len(contnr)))*contsum
    return contav


# In[ ]:

# run_container takes a textfile and gives two numpy arrays.
# Array[1]: Each row is the average of the vector rep 
#           of each word in a bag of size contsize. 
#           This is the input for the NNet
# Array[2]: Each row is the vector rep of the middle word in 
#           each container. This is the target of the NNet
# Bags are continuously constructed until linenumber "lines" is reached

def run_container(tfile, dictcut, contsize, lines):
    with open(tfile) as f:
        
        contnr=[] # will continuously contain the bag of words
        linenum=0 # counter for lines in the training data set
        dlen=len(dictcut) # length of the dictionary, and length of the vector for each word
        inarr=np.empty((0,dlen))
        outarr=np.empty((0,dlen))
        
        for line in f:
            if linenum==lines:
                break
            line=(line.lower()).translate(None, string.punctuation)
            for word in nltk.word_tokenize(line):
                if word in dictcut:
                    contnr.append(dictcut[word]) # add next word into container
                if len(contnr)==contsize: # when container is full
                    linenum+=1
                    inarr=np.vstack((inarr,container_sum(contnr,dlen))) # add av of bag to input
                    # note contsize is odd so contsize/2 is integer and the middle of the container
                    outarr=np.vstack((outarr,unit_vector(contnr[contsize/2],dlen))) # add target word to output
                    contnr=contnr[1:] # delete first word in container 
                
        return [inarr,outarr]

def w2vdatafromtext(tfile,contsize,trainlines,freqcutoff):
    textdict=textfile_worddict(tfile,lines=trainlines)
    textdictcut=dict_cutoff(textdict,freqcutoff)
    w2vdata=run_container(tfile,textdictcut,contsize,lines=trainlines)
    return [textdictcut,w2vdata[0],w2vdata[1]]


# In[ ]:

# W2Vdata=w2vdatafromtext("textfilesmall.txt",contsize=5,trainlines=10,freqcutoff=1)
# print W2Vdata

