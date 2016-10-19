
# coding: utf-8

# In[1]:

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

# In[30]:

#------------------------------------------
def linecount(tfile):
    with open(tfile) as f:
        i=0
        for line in f:
            i+=1
    return i 

#------------------------------------------
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


#------------------------------------------
# run_container takes a textfile and a dictionary
# and outputs two numpy arrays:
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
        inarr=[]
        outarr=[]
        
        for line in f:
            if linenum>lines:
                break
            linenum+=1
            line=(line.lower()).translate(None, string.punctuation)
            for word in nltk.word_tokenize(line):
                if word in dictcut:
                    contnr.append(dictcut[word]) # add next word into container
                if len(contnr)==contsize: # when container is full
                    inarr.append(contnr) # add bag of words to input
                    # note contsize is odd so contsize/2 is integer and the middle of the container
                    outarr.append([contnr[contsize/2]]) # add target word to output
                    contnr=contnr[1:] # delete first word in container 
                
        return [inarr,outarr]

#------------------------------------------
# This function takes a textfile and some parameters
# and outputs two numpy arrays and a word dictionary for W2V, 
# see run_container fordetails of output

def w2vdatafromtext(tfile,contsize,trainlines,freqcutoff):
    textdict=textfile_worddict(tfile,lines=trainlines)
    textdictcut=dict_cutoff(textdict,freqcutoff)
    w2vdata=run_container(tfile,textdictcut,contsize,lines=trainlines)
    return [textdictcut,w2vdata[0],w2vdata[1]]


# In[34]:

w2vdatafromtext("delorme.txt", contsize=7, trainlines=100000,freqcutoff=1)


# In[ ]:



