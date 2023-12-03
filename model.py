import numpy as np
import scipy as scp
import nltk
import math
import string, re
import tensorflow as tf
from random import choice, shuffle
from numpy import array

import tensorflow_hub as hub

import sys, os, unicodedata
sys.path.insert(0, './skip-thoughts-master')
import skipthoughts
import time

import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize

from scipy import sparse
from termcolor import colored

from sklearn.cluster import KMeans

#word2vec encoding, sentece length, latent s hashing

def remove_control_chart(s):
    return re.sub(r'\\x..', '', s)

#Extract sentences from the text document
def extract_sentences(document):
    #Read file into sentences
    doc_content = open(document,'r').read()
    doc_content = doc_content.replace("\r",".")
    doc_content = doc_content.replace("\n",".")
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentence_set = sent_detector.tokenize(doc_content.strip())

    #Remove puctuation marks from the sentences
    sentence_set_without_punctuation = []
    table = str.maketrans({key: None for key in string.punctuation})
    for sentence in sentence_set:
        #sentence = unicodedata.normalize('NFKD', sentence).encode('ascii', 'ignore')
        sentence = re.sub("[\(\[].*?[\)\]]", "", sentence)
        sentence = re.sub(r'[^\x00-\x7F]+','', sentence)
        #print(sentence.translate(table))
        sentence = sentence.translate(table)
        sentence = sentence.replace("etal", "")
        sentence = sentence.rstrip()

        if sentence != '' and len(sentence.split())>8:
            sentence_set_without_punctuation.append(sentence)

    return sentence_set_without_punctuation;

#Remove stop words from the sentence tokens
def remove_stop_words(sentence):
    word_tokens = word_tokenize(sentence)
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence


def TFKMeansCluster(vectors, noofclusters):
 
    noofclusters = int(noofclusters)
    assert noofclusters < len(vectors)
 
    #Find out the dimensionality
    dim = len(vectors[0])
 
    #Will help select random centroids from among the available vectors
    vector_indices = list(range(len(vectors)))
    shuffle(vector_indices)
 
    graph = tf.Graph()
 
    with graph.as_default():
 
        #SESSION OF COMPUTATION
 
        sess = tf.Session()

        centroids = [tf.Variable((vectors[vector_indices[i]]))
                     for i in range(noofclusters)]

        centroid_value = tf.placeholder("float64", [dim])
        cent_assigns = []
        for centroid in centroids:
            cent_assigns.append(tf.assign(centroid, centroid_value))
 

        assignments = [tf.Variable(0) for i in range(len(vectors))]

        assignment_value = tf.placeholder("int32")
        cluster_assigns = []
        for assignment in assignments:
            cluster_assigns.append(tf.assign(assignment,
                                             assignment_value))

        mean_input = tf.placeholder("float", [None, dim])

        mean_op = tf.reduce_mean(mean_input, 0)

        v1 = tf.placeholder("float", [dim])
        v2 = tf.placeholder("float", [dim])
        euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2)))

        centroid_distances = tf.placeholder("float", [noofclusters])
        cluster_assignment = tf.argmin(centroid_distances, 0)
 
        init_op = tf.initialize_all_variables()
 
        #Initialize all variables
        sess.run(init_op)

        noofiterations = 100
        for iteration_n in range(noofiterations):
 

            for vector_n in range(len(vectors)):
                vect = vectors[vector_n]

                distances = [sess.run(euclid_dist, feed_dict={
                    v1: vect, v2: sess.run(centroid)})
                             for centroid in centroids]
                #Now use the cluster assignment node, with the distances
                #as the input
                assignment = sess.run(cluster_assignment, feed_dict = {
                    centroid_distances: distances})
                #Now assign the value to the appropriate state variable
                sess.run(cluster_assigns[vector_n], feed_dict={
                    assignment_value: assignment})
 
            ##MAXIMIZATION STEP

            for cluster_n in range(noofclusters):
                #Collect all the vectors assigned to this cluster
                assigned_vects = [vectors[i] for i in range(len(vectors))
                                  if sess.run(assignments[i]) == cluster_n]
                #Compute new centroid location
                new_location = sess.run(mean_op, feed_dict={
                    mean_input: array(assigned_vects)})
                #Assign value to appropriate variable
                sess.run(cent_assigns[cluster_n], feed_dict={
                    centroid_value: new_location})
 
        #Return centroids and assignments
        centroids = sess.run(centroids)
        assignments = sess.run(assignments)
        return centroids, assignments


#Create a dictionary to store document frequency
def create_df_dictionary(sentence_set):
    df_dictionary = {}
    for sentence in sentence_set:
        word_list = remove_stop_words(sentence)
        word_list = set(word_list)
        for word in word_list:
            if word not in df_dictionary:
                df_dictionary[word] = 1
            else:
                 df_dictionary[word] += 1
    return df_dictionary

#Get tfIdf value of a particular word in the dictionary
def get_tfIdf(sentence_set, df_dictionary, tokens, word):
    tfIdf = 0
    #term-frequency
    tf_word = float(tokens.count(word)) / len(tokens)
    #Inverse document frequency
    idf_word = math.log(float(len(sentence_set))/(1+df_dictionary[word]))
    #Tf-Idf value
    tfIdf = tf_word*idf_word
    return tfIdf

#Construct the source matrix A needed for SVD decomposition
def construct_source_matrix(sentence_set,df_dictionary, word_list_in_document):
    #Create SVD source array
    source_array = []
    for sentence in sentence_set:
        sentence_array = [0]*len(word_list_in_document)
        tokens = remove_stop_words(sentence)
        token_set = set(tokens)
        for word in token_set:
            sentence_array[word_list_in_document.index(word)] = get_tfIdf(sentence_set, df_dictionary, tokens, word)
        source_array.append(sentence_array)
    ###print(source_array)

    #Convert array to numpy sparse matrix
    source_array = np.matrix(source_array)
    source_array = source_array.getT()
    ###source_array_sparse = sparse.csr_matrix(source_array)

    return source_array

#Run SVD decomposition
def run_SVD(source_array):
    u,s,vH = np.linalg.svd(source_array, full_matrices=False)
    return u,s,np.flipud(vH)

#Gong-liu approach for sentence extraction
def get_sentences_gongLiu(sentence_set, vH, num_of_sentences, paper_directory):
    print(colored('Important sentences by gong-liu approach | Num of sentences : ' + str(num_of_sentences), 'red'))
    sentence_index_arr = []
    num_of_sentences = min(num_of_sentences, len(sentence_set))
    for i in range(num_of_sentences):
        temp_arr = vH[i,:]
        sentence_index = temp_arr.argmax(axis=1)[0,0]
        sentence_index_arr.append(sentence_index)
    sentence_index_arr = sorted(set(sentence_index_arr))

    f = open(paper_directory, "a")
    print(paper_directory)
    num = 0
    for i in sentence_index_arr:
        print(str(num) + ' :' + sentence_set[i])
        f.write('->' + sentence_set[i] + "\n")
        num+=1
    f.close()
    print('-----------')


def get_sentences_stanberg(sentence_set, s, vH, num_of_sentences, paper_directory):
    print(colored('Important sentences by Steinberger approach | Num of sentences : ' + str(num_of_sentences), 'red'))
    sentences_length = []
    vH = vH.getT()
    for i in range(np.size(vH,0)):
        val = 0.0
        for j in range(np.size(vH,1)):
            val += s[j]*s[j]*vH[i,j]*vH[i,j]
        sentences_length.append(math.sqrt(val))
    sentences_length_copy = sentences_length.copy()
    sentences_length.sort()
    sentences_length.reverse()

    sentence_index_arr = []
    num_of_sentences = min(num_of_sentences, len(sentence_set))
    for i in range(num_of_sentences):
        sentence_index_arr.append(sentences_length_copy.index(sentences_length[i]))
        #print(str(i) + ' : ' + sentence_set[sentences_length_copy.index(sentences_length[i])])
    sentence_index_arr = sorted(set(sentence_index_arr))
    num = 0
    f = open(paper_directory, "a")
    for i in sentence_index_arr:
        print(str(num) + ' :' + sentence_set[i])
        f.write("->" + sentence_set[i] + "\n")
        num+=1
    f.close()

def get_sentences_topic_method(sentence_set, s, vH, num_of_sentences, num_of_sen_per_concept, paper_directory):
    print(colored('Important sentences by topic approach | Num of sentences : ' + str(num_of_sentences), 'red'))
    avg_sentence_score = []
    for i in range(np.size(vH,0)): #rows
        avg_val = 0.0
        for j in range(np.size(vH, 1)):
            avg_val += vH[i,j]
        avg_sentence_score.append(avg_val/np.size(vH,1))

    for i in range(np.size(vH,0)):
        score = avg_sentence_score[i]
        for j in range(np.size(vH,1)):
            if vH[i,j]<=score:
                vH[i,j] = 0

    concept_cross_matrix = [[0]*(np.size(vH, 0)+1) for x in range(np.size(vH, 0))]
    strength_arr = []
    for i in range(np.size(vH,0)):
        total_strength = 0
        for j in range(np.size(vH,0)):
            strength_val = 0.0
            for k in range(np.size(vH,1)):
                if vH[i,k]!=0.0 and vH[j,k]!=0.0:
                    strength_val += vH[i,k] + vH[j,k]

            concept_cross_matrix[i][j] = strength_val
            total_strength += strength_val

        concept_cross_matrix[i][np.size(vH, 0)] = total_strength
        strength_arr.append(total_strength)

    strength_arr_copy = strength_arr.copy()
    strength_arr.sort()
    strength_arr.reverse()
    sentence_index_arr = []
    num = num_of_sentences
    for i in strength_arr:
        conc_index = strength_arr_copy.index(i)
        conc_num = 0
        for j in range(np.size(vH,1)): #1->columns
            if vH[conc_index, j]!=0 and conc_num<=num_of_sen_per_concept:
                sentence_index_arr.append(j)
                num_of_sentences += -1
                conc_num+=1
    sentence_index_arr = sorted(set(sentence_index_arr))
    num = 0
    f = open(paper_directory, "a")
    for i in sentence_index_arr:
        print(str(num) + ' : ' + sentence_set[i])
        f.write("->" + sentence_set[i] + "\n")
        num+=1
    f.close()

def load_embedding_model(embedding_type):
    if embedding_type=='skip_thought':
      return skipthoughts.load_model()

def summary_for_sentence_set(sentence_set, paper_directory):
    print('Length of set : ' + str(len(sentence_set)))
    df_dictionary = create_df_dictionary(sentence_set)

    #List of words in the document
    word_list_in_document = list(df_dictionary.keys())

    #Construct the SVD source sparse matrix
    source_array = construct_source_matrix(sentence_set, df_dictionary, word_list_in_document)

    u,s,vH = run_SVD(source_array)

    if len(sentence_set)<7:
        get_sentences_gongLiu(sentence_set, vH, 3, paper_directory)

    elif(len(sentence_set)<12):
        get_sentences_stanberg(sentence_set, s, vH, 3, paper_directory)

    else:
        get_sentences_topic_method(sentence_set, s, vH, 4, 1, paper_directory)

def summmary_for_section(section_text, encoder, paper_directory):
   sentence_set = extract_sentences(section_text) 
   if len(sentence_set)==0:
       print('')
   elif len(sentence_set)<4:
      summary_for_sentence_set(sentence_set, paper_directory)
   else:   
      encoded = encoder.encode(sentence_set)
      n_clusters = np.ceil(len(encoded)**0.5)
      kmeans = KMeans(n_clusters=n_clusters)
      kmeans = kmeans.fit(encoded)
   
      mydict = {i: np.where(kmeans.labels_ == i)[0] for i in range(int(kmeans.n_clusters))}
   
      sentence_dict = {}
      for cluster_num in mydict.keys():
       sentence_dict[cluster_num] = []
       for i in mydict[cluster_num]:
         sentence_dict[cluster_num].append(sentence_set[i])
      #print(sentence_dict)
   
      for cluster_num in sentence_dict.keys():
        summary_for_sentence_set(sentence_dict[cluster_num], paper_directory)


def summmary_for_section_2(section_text, embed, paper_directory):
    sentence_set = extract_sentences(section_text)

    if len(sentence_set)==0:
        print('')
    elif len(sentence_set)<=4:
        summary_for_sentence_set(sentence_set, paper_directory)
    else:
        num_of_clusters = round(0.1147*len(sentence_set) + 0.5662)
        embedding_arr = []
        cluster_dict = {}
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            message_embeddings = session.run(embed(sentence_set))
            for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
                embedding_arr.append(message_embedding)
        embedding_arr = np.array(embedding_arr)
        centroids, assignments = TFKMeansCluster(embedding_arr, num_of_clusters)
        for i in range(len(assignments)):
            if assignments[i] not in cluster_dict:
                cluster_dict[assignments[i]] = [sentence_set[i]]
            else:
                cluster_dict[assignments[i]].append(sentence_set[i])

        for i in cluster_dict.keys():
            summary_for_sentence_set(cluster_dict[i], paper_directory)

        

def main():

    #Scikit learn skip thought implementation
    model = skipthoughts.load_model()
    encoder = skipthoughts.Encoder(model)
    paper_directories = os.listdir("./Parsed_Papers_Test/")
    paper_directories = ["./Parsed_Papers_Test/" + x for x in paper_directories]
    for paper_directory in paper_directories:
        #print(paper_directory)
        paper_name = paper_directory.split("/")[2]
        file = open("./" + paper_name + ".txt", "w")
        file.close()
        for root, dirs, files in os.walk(paper_directory):
            files = [paper_directory + "/" + x for x in files]
            for section in files:
                print(section)
                summmary_for_section(section, encoder, "./" + paper_name + ".txt")
                print('_________')       
    

    #Tensorflow Universl encoder implementation
    embed = hub.Module("./tmp/moduleA")
    paper_directories = os.listdir("./Parsed_Papers_Test/")
    paper_directories = ["./Parsed_Papers_Test/" + x for x in paper_directories]

    for paper_directory in paper_directories:
        #print(paper_directory)
        paper_name = paper_directory.split("/")[2] 
        file = open("./" + paper_name + ".txt", "w")
        file.close()
        for root, dirs, files in os.walk(paper_directory):
            files = [paper_directory + "/" + x for x in files]
            for section in files:
                print(section)
                summmary_for_section_2(section, embed, "./" + paper_name + ".txt")
                print('_________')
    
    
    

if __name__ == '__main__':
    main()
