import csv
import ast
import re
import itertools 
import time
import os
import io
import gensim
import math
import sys
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.word2vec import Word2Vec
from scipy import spatial
import sent2vec
import multiprocessing
from glove import Corpus, Glove

stop_words1 = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]


stopfile = 'stopwords-en.txt'
def load_stop_words():
	return set(line.strip() for line in open(stopfile, errors='ignore'))

stop_words2 = load_stop_words()


#from gensim.models import KeyedVectors#BioSentVec_PubMed_MIMICIII-bigram_d700.bin
#print(1)
#word2vec_model = KeyedVectors.load_word2vec_format('/mnt/local/hdd/Javadr/trained_models/pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin', binary=True)
#word2vec_model = Word2Vec.load('/mnt/local/hdd/Javadr/trained_models/Word2Vec/200/Word2Vec_Model.bin')

#print(2)



#load sent2vec
print(1)
sent2vec_model = sent2vec.Sent2vecModel()
sent2vec_model.load_model('/mnt/local/hdd/Javadr/trained_models/Sen2Vec/700/sent2vec_wiki_bigrams.bin')#sent2vec_twitter_bigrams.bin
print(2)

def remove_special_characters(text):
	return re.sub("([{}@\"$%+=<>,&\\\/*'\"]|\d)", ' ', text)

def pre_process(content, dd = 0):
	content = content.replace('\n',' ')
	content = content.replace('	',' ')
	content = content.lower()
	content=remove_special_characters(content)
	
	while '  ' in content:
		content=content.replace('  ',' ')
	word_tokens = content.split(' ')
	word_tokens = [w for w in word_tokens if w not in stop_words1 and w not in stop_words2 and len(w)>1]
	if(dd ==1):
		return word_tokens
	return ' '.join(word_tokens)

def wordTovec(sentence):
	global word2vec_model
	vocab = list(word2vec_model.wv.vocab.keys())
	word_tokens = sentence.split(' ') # or  word_tokens = word_tokenize(sentence)
	i =0
	v = [0]*size0fvector
	for word in word_tokens:
		if word in vocab:
			if i ==0:
				#print('yes')
				v = word2vec_model[word]
			else:
				v = np.add(v,word2vec_model[word])
			i = i+1
	#print(v)
	#print(sentence)	
	#x = int(input('Enter a number:'))
	return v
	
def word2vec_Local(abstract):			
	documents = [pre_process(abstract,1)]
	model = gensim.models.Word2Vec(
		documents,
		size=20,
		window=2,
		min_count=2,
		workers=max(1, multiprocessing.cpu_count() - 1))
	model.train(documents, total_examples=len(documents), epochs=5)
	#print(list(model.wv.vocab.keys()))
	return model


def wordTovec2(sentence):
	global word2vec_model
	vocab = word2vec_model.keys()
	word_tokens = sentence.split(' ') # or  word_tokens = word_tokenize(sentence)
	i =0
	v = [0]*size0fvector
	for word in word_tokens:
		if word in vocab:
			if i ==0:
				#print('yes')
				v = word2vec_model[word]
			else:
				v = np.add(v,word2vec_model[word])
			i = i+1
	#if i ==0:
	#	print(v,word_tokens)
	#print(v)
	#print(sentence)	
	#x = int(input('Enter a number:'))
	return v

def glove_Local(abstract):			
	documents = [pre_process(abstract,1)]

	corpus_model = Corpus()
	corpus_model.fit(documents, window=6)

	glove = Glove(no_components=20, learning_rate=0.005)
	glove.fit(corpus_model.matrix, epochs=5, no_threads=max(1, multiprocessing.cpu_count() - 1))
	glove.add_dictionary(corpus_model.dictionary)
	#print(list(glove.dictionary.keys()))
	return glove

def loadGloveModel(gloveFile):
	f = io.open(gloveFile, 'r', encoding='utf-8', newline='\n', errors='ignore')
	f.readline()
	glob_model = {}
	for line in f:
		splitLine = line.rstrip().split()
		word = splitLine[0]
		embedding = np.array(splitLine[1:])
		glob_model[word] = embedding.astype(np.float)
	return glob_model

def sentTovec(sentence):
	global sent2vec_model
	return sent2vec_model.embed_sentence(sentence)

def similarity(v1,v2):
	return (1 - spatial.distance.cosine(v1, v2))

paths = ['Acute_Threat_Fear.csv','Loss.csv','Arousal.csv','Circadian_Rhythms.csv','Frustrative_Nonreward.csv','Potential_Threat_Anxiety_.csv','Sleep_Wakefulness.csv','Sustained_Threat.csv']
constructs = ['acute threat fear','loss','arousal','circadian rhythms','frustrative nonreward','potential threat anxiety','sleep wakefulness','sustained threat']
construct_extra = ['loss is a state of deprivation of a motivationally significant con-specific, object, or situation','acute threat fear is Activation of the brain’s defensive motivational system to promote behaviors that protect the organism from perceived danger','frustrative nonreward is reactions elicited in response to withdrawal prevention of reward','potential threat anxiety is activation of a brain system in which harm may potentially occur but is distant, ambiguous, or uncertain in probability, characterized by a pattern of responses such as enhanced risk assessment ','Sleep and wakefulness are endogenous, recurring, behavioral states that reflect coordinated changes in the dynamic functional organization of the brain','sustained threat is an aversive emotional state caused by prolonged exposure to internal or external condition, state, or stimuli that are adaptive to escape or avoid.','Arousal is a continuum of sensitivity of the organism to stimuli, external and internal.','Circadian Rhythms are endogenous self-sustaining oscillations that organize the timing of biological systems to optimize physiology and behavior, and health']
################
size0fvector=700

'''
print(1)
word2vec_model = loadGloveModel('/mnt/local/hdd/Javadr/trained_models/Word2Vec/glove/glove.840B.300d.txt')#glove.840B.300d.txt
print(2)
'''

#/home/javadr/code/RdoC
#/mnt/local/hdd/Javadr/trained_models
################
avg=0.0
for vv in range(len(paths)):
	path = paths[vv]

	csvfile = open(path)
	readCSV = csv.reader(csvfile, delimiter=',')
	print(path)
	i =0
	count = 0
	#####################
	#word2vec_model = [] # for local
	#####################
	for row in readCSV:
		i = i+1
		#print(row)
		if i ==1:
			continue
	
		pmid = row[0].lower()
		title = row[1].lower()
		abstract = row[2].lower()
		relevants = list(ast.literal_eval(row[3].lower()))
		sentences = re.split(r"[.!?]", abstract)
		#######################################
		#word2vec_model = glove_Local(abstract)#for local
		#######################################
		for j in range(len(sentences)):
			if j !=0:
				sentences[j] = sentences[j][1:]
	
		del sentences[-1]
		sim_dict={}
		################
		original = constructs[vv] #constructs[vv]
		################
		for sentence in sentences:
			######################################################################################################
			sim_dict [sentence] = similarity(sentTovec(pre_process(sentence)),sentTovec(original)) 
			######################################################################################################
		sim_dict = {k:v for k, v in sorted(sim_dict.items(), key=lambda t: t[1],reverse=True)}
	
		sentence = ''
		for key, value in sim_dict.items():
			if len(pre_process(key)) <= 70:
				continue
			sentence = pre_process(key)
			break
	
	
		_max_ = 0.0
		for item in relevants:
			if item.endswith('.'):
				item = item[0:len(item)-1]
			cc =0.0		
			sentence_tokens = sentence.split(' ')
			item_tokens = pre_process(item).split(' ')
			for token in sentence_tokens:
				if token in item_tokens:
					cc = cc +1
			cc = cc/len(sentence_tokens)
			if (_max_<cc):
				_max_ = cc
		print(i,_max_)
		if _max_ >= 0.6:
			count = count + 1
		print(sentence)
		#x = int(input('Enter a number:'))
	print(constructs[vv]+" : "+ str(count/(i-1)))
	avg = avg + count/(i-1)
print('Total: '+ str(avg/8))
