
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

from scipy.signal import argrelextrema

import pandas as pd

# Import matplotlib
import matplotlib.pyplot as plt

# Import useful Python standard libraries

import numpy as np
import ast
import glob
import os
import random

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens




genre_strings = [  'Action', 'Adventure', 'Animation', 'Biography', 'Comedy',
            'Crime', 'Drama', 'Family', 'Fantasy', 'Film-Noir', 'History',
            'Horror', 'Music', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
            'Short', 'Sport', 'Thriller', 'War', 'Western']

action, adventure, animation, biography, comedy, crime, drama, family, fantasy, noir, history, horror, music, musical, mystery, romance, scifi, short, sport, thriller, war, western = [[] for _ in range(22)]

genres = [ action, adventure, animation, biography, comedy, crime, drama,
          family, fantasy, noir, history, horror, music, musical, mystery,
          romance, scifi, short, sport, thriller, war, western]

for i, genre in enumerate(genres):
    path = f'/Users/jonathansalama/Documents/PrincetonYear4/TRA301/final_project/imsdb_raw_nov_2015/{genre_strings[i]}'
    for filename in glob.glob(os.path.join(path, '*.txt')):
        # print(filename)
        with open(filename, 'r') as f: # open in readonly mode
            genre.append((os.path.basename(filename), f.read()))

            

action2, adventure2, animation2, biography2, comedy2, crime2, drama2, family2, fantasy2, noir2, history2, horror2, music2, musical2, mystery2, romance2, scifi2, short2, sport2, thriller2, war2, western2 = [[] for _ in range(22)]

genres2 = [ action2, adventure2, animation2, biography2, comedy2, crime2, drama2,
          family2, fantasy2, noir2, history2, horror2, music2, musical2, mystery2,
          romance2, scifi2, short2, sport2, thriller2, war2, western2]


lines = []
with open('/Users/jonathansalama/Documents/PrincetonYear4/TRA301/final_project/output.txt', 'r') as f:
  for line in f:
    lines.append(line.strip())
      

for l, line in enumerate(lines):
  if '.txt' in line:
    for g, genre in enumerate(genres):
      titles = [gen[0] for gen in genre]
      scripts = [gen[1] for gen in genre]
      if line in titles:
        y_vals = ast.literal_eval(lines[l+2])
        if (line, y_vals, scripts[titles.index(line)]) not in genres2[g]:
          genres2[g].append((line, y_vals, scripts[titles.index(line)]))

    # movie_scores.append((line, ast.literal_eval(lines[l+2])))

# print(genres)

lines = []
with open('/Users/jonathansalama/Documents/PrincetonYear4/TRA301/final_project/genrevectors.txt', 'r') as f:
  for line in f:
    lines.append(line.strip())


vector_map = {}

for l, line in enumerate(lines):
  if '.txt' in line:
    vec = ast.literal_eval(lines[l+1])
    if line not in vector_map:
        vector_map[str(line)] = vec


for g, genre in enumerate(genres2):

  best_lines = []

  for m, movie_score in enumerate(genre):
    y = np.array(movie_score[1])
    x = np.arange(0, len(movie_score[1]))

    if len(x) == 0:
      continue


    y_end = [yv for yv in y]
    max_end = sorted(y_end[-5:], key=abs)[-1]
    y[-5:]     =  [(yv*2+max_end)/3 for yv in y_end[-5:]]
    y_end[-5:] =  [(yv*2+max_end)/3 for yv in y_end[-5:]]

    try:
      z1 = np.polyfit(x, y_end, 4)
      z2 = np.polyfit(x, y_end, 6)
      z3 = np.polyfit(x, y_end, 8)
      z4 = np.polyfit(x, y_end, 10)
      z5 = np.polyfit(x, y_end, 12)
    except:
      continue

    f1 = np.poly1d(z1)
    f2 = np.poly1d(z2)
    f3 = np.poly1d(z3)
    f4 = np.poly1d(z4)
    f5 = np.poly1d(z5)

    # calculate new x's and y's
    x_new = np.linspace(x[0], x[-1], 300)
    y_new1 = f1(x_new)
    y_new2 = f2(x_new)
    y_new3 = f3(x_new)
    y_new4 = f4(x_new)
    y_new5 = f5(x_new)

    movie_vector = [y_new1, y_new2, y_new3, y_new4, y_new5]

    genres2[g][m] = (movie_score[0], movie_vector, movie_score[2])



cluster1 = ['Music', 'Musical', 'Comedy', 'Romance', 'Sport', 'Short', 'Family', 'Fantasy', 'Drama']
cluster2 = ['Horror', 'Thriller', 'Crime', 'Film-Noir', 'Sci-Fi', 'Mystery', 'Action', 'Western', 'Adventure', 'Animation']
cluster3 = ['Documentary', 'Biography', 'History', 'War']


movie_info = []
movie_names = []
all_movies_scenes = []

for i, genre in enumerate(genres2):
  gen_name = genre_strings[i]

  for s, script in enumerate(genre):
    if script[0] not in movie_names:

        lines = script[2].splitlines()
        scenes = []
        curr_scene = ''
        in_scene = False

        characters = []

        for line in lines:
            line = line.strip()
            if line == '':
                continue

            if line.isupper() and ('INT' in line or 'EXT' in line or 'CUT TO' in line):
                in_scene = True
                if curr_scene != '':
                    scenes.append(curr_scene)
                    curr_scene = line + ' -- '
                    continue

            elif line.isupper() and len(line.split()) < 3:
                curr_char = line.split()[0]
                # if "CONTINUED" in curr_char or "CONT'D" in curr_char:
                #   characters.append(characters[-1])
                # else:
                characters.append(curr_char)
                line = line.strip() + ':'

            if in_scene:
                curr_scene += (line + ' ')
                # print(curr_scene)

        if curr_scene != '':
            scenes.append(curr_scene)
            curr_scene = ''

        if len(scenes) < 5:
            continue
        

        all_movies_scenes.extend(scenes)

        movie_info.append((script[0], script[1], script[2], [gen_name], scenes))
        movie_names.append(script[0])
    
    else:
        curr_index = movie_names.index(script[0])
        curr_info = list(movie_info[curr_index])
        curr_info[3].append(gen_name)
        movie_info[curr_index] = tuple(curr_info)


# for part in movie_info[0]:
#     print(part[:150])



##########################################################################################################################################

# print("starting doc2v training")
# documents = [TaggedDocument(tokenize_text(doc), [i]) for i, doc in enumerate(all_movies_scenes)]
# model = Doc2Vec(documents, vector_size=50, window=2, min_count=1, workers=4, epochs=20)


fname = '/Users/jonathansalama/Documents/PrincetonYear4/TRA301/final_project/script_doc2vec_model'

# model.save(fname)
d2v_model = Doc2Vec.load(fname)  # you can continue training with the loaded model!

print('finished d2v training')


##########################################################################################################################################


def getSemanticEmbeddings(data, sentiment_list, bert_model, d2v_model, scenes_lists):

    climax_data = []

    d2v_embeddings = []

    for s, script in enumerate(data):
        lines = script.split('\n')
        script_size = len(lines)
        # print(script_size)
        sentiment_vec = sentiment_list[s]

        scene_list = scenes_lists[s]

        average_line = np.mean(sentiment_vec, axis=0)

        # for local maxima
        maxima = argrelextrema(average_line, np.greater)[0].tolist()

        # for local minima
        minima = argrelextrema(average_line, np.less)[0].tolist()

        try:
            maxmin = sorted(maxima + minima)
        except Exception as e:
            print(maxima, minima)
            print(e)
            raise TypeError


        max_scene_val = maxmin[-1] / len(average_line)

        # print("max:", max_scene_val)

        max_line_location = round(script_size * max_scene_val)
        max_scene_location = round(len(scene_list) * max_scene_val)


        line_buffer = round(script_size / 40)

        climax_lines = lines[ (max_line_location - line_buffer) : (max_line_location + line_buffer) ]

        climax_scenes = scene_list[ (max_scene_location - 5) : (max_scene_location + 5)]


        climax_data.append('\n'.join(climax_lines))

        curr_d2v = np.mean([d2v_model.infer_vector(tokenize_text(scene)) for scene in climax_scenes], axis=0)

        # print("CURR D2V shape:", curr_d2v.shape)

        d2v_embeddings.append(curr_d2v)



    bert_embeddings = bert_model.encode(climax_data)

    return bert_embeddings, d2v_embeddings






pd.options.display.width = 0

all_posts = []
all_posts_names = []

for movie in movie_info:
    if movie[0] not in all_posts_names:
        all_posts.append(movie)
        all_posts_names.append(movie[0])
    else:
        print(movie[0])


# engaged_sentences = ['missionimpossible.txt', 'oblivion.txt', 'indianajonesandthelastcrusade.txt']

print(len(all_posts))


content_matrix = []

for i, movie in enumerate(all_posts):
        content_matrix.append([movie[0], movie[2], movie[1], vector_map[str(movie[0])], movie[3], movie[4]])
        # print([categories[i], sentence], file=f)


df = pd.DataFrame(content_matrix, columns=[
                    'Title', 'Description', 'Sentiment', 'GenreVec', 'Genres', 'Scenes'])


print(df.head(10))

titles = np.array(df['Title'])
content_data = np.array(df["Description"])
sentiment_list = df['Sentiment'].tolist()
genre_vec_list = df['GenreVec'].tolist()
genres_label_list = df['Genres'].tolist()
scenes_lists = df['Scenes'].tolist()


model = SentenceTransformer('all-MiniLM-L6-v2')
bert_embeddings, d2v_embeddings = getSemanticEmbeddings(content_data, sentiment_list, model, d2v_model, scenes_lists)
# bert_embeddings = [[0 for _ in range(384)] for _ in all_posts]
# d2v_embeddings = [[0 for _ in range(50)] for _ in all_posts]

bert_vectors = np.array(bert_embeddings)
d2v_vectors = np.array(d2v_embeddings)

print()
print("done with bert and d2v")
print()



#######################################
from sklearn import model_selection, preprocessing, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

from sklearn import metrics
from sklearn import ensemble, neural_network, neighbors
from sklearn.multioutput import MultiOutputClassifier

import tensorflow as tf
from keras.layers import  Dropout, Dense, Flatten, Embedding
from keras.models import Sequential
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
# from tensorflow.keras.utils import to_categorical


from keras.optimizers import gradient_descent_v2, adam_v2



labeled_movies = []


for t, title in enumerate(titles):
    labeled_movies.append((title, content_data[t], sentiment_list[t], genre_vec_list[t], bert_vectors[t], d2v_vectors[t], genres_label_list[t]))

# random.shuffle(labeled_movies)

# print(labeled_movies[0])

labels, texts = [], []
for i, movie in enumerate(labeled_movies):
    label = movie[6]
    new_label = []
    for gr in genre_strings:
        if gr in label:
            new_label.append(1)
        else:
            new_label.append(0)
    labels.append(new_label)
    print("LABEL:", new_label)
    texts.append(movie[:6])


for t, text in enumerate(texts):

    svecs = text[2]
    svecs = np.mean(svecs, axis=0)
    gvecs = np.asarray(text[3])
    bvecs = np.asarray(text[4])
    dvecs = np.asarray(text[5])

    # print("SVECS:", svecs.shape)
    # print("GVECS", np.shape(gvecs))
    # print("BVECS", np.shape(bvecs))
    # print("DVECS", np.shape(dvecs))


    

    texts[t] = np.concatenate([svecs, gvecs, bvecs, dvecs], axis=0)
    if texts[t].shape != (744,):
        print("array shape:", texts[t].shape)


print("DATASET SIZE:", len(texts))

# create a dataframe using texts and lables
trainDF = pd.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels

# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])



def get_score(predictions):
    results = []
    easy_results = []
    rs = [[] for _ in range(22)]

    for ind, pred in enumerate(predictions):
        actual = valid_y[ind]
        score = 0.0
        easy_score = 0.0

        cluster1score = 0
        cluster2score = 0
        cluster3score = 0

        for j in range(22):
            if pred[j] == actual[j]:
                score += 1/22
                easy_score += 1/22
                rs[j].append(1.0)

                if genre_strings[j] in cluster1:
                    cluster1score += 1
                if genre_strings[j] in cluster2:
                    cluster2score += 1
                if genre_strings[j] in cluster3:
                    cluster3score += 1

            else:
                score -= 1/22
                rs[j].append(0.0)

        # if sum(actual) < sum(pred):
        #     score = max(0.0, (score-0.25)*(sum(pred)-sum(actual)))

        max_cluster = None
        max_cluster_score = max(cluster1score, cluster2score, cluster3score)
        if max_cluster_score == cluster1score:
           max_cluster = cluster1
        elif max_cluster_score == cluster2score:
            max_cluster = cluster2
        elif max_cluster_score == cluster3score:
            max_cluster = cluster3


        for k in range(22):
            if pred[k] == 1 and actual[k] == 0:
                curr_genre = genre_strings[k]
                if curr_genre not in max_cluster:
                    for cluster in [cluster1, cluster2, cluster3]:
                        if curr_genre in cluster:
                            score = min(score, 0.5 * (cluster.index(curr_genre) / len(cluster)))
                

        
        results.append(score)
        easy_results.append(easy_score)

    r_score = sum(results)/len(results)
    e_score = sum(easy_results)/len(easy_results)

    
    individual_scores = []
    for rsc in rs:
        if rsc == []:
            individual_scores.append(0.0)
        else:
            individual_scores.append(sum(rsc)/len(rsc))


    return r_score, individual_scores, e_score


def get_precision_recall(predictions, valid):

    true_p = 0
    false_p = 0
    false_n = 0


    for ind, pred in enumerate(predictions):
        actual = valid[ind]
        score = 0.0
        easy_score = 0.0

        for j in range(22):
            if pred[j] == 1 and actual[j] == 1:
                true_p += 1
            elif pred[j] == 1 and actual[j] == 0:
                false_p += 1
            elif pred[j] == 0 and actual[j] == 1:
                false_n += 1

    precision = (true_p)/(true_p + false_p)
    recall = (true_p)/(true_p + false_n)

    return precision, recall






    


















def train_model_pipeline(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):

    text_clf = classifier

    text_clf = MultiOutputClassifier(classifier, n_jobs=-1)

    text_clf.fit(feature_vector_train, label)

    predictions = text_clf.predict(feature_vector_valid)
    print(len(predictions))
    
    return predictions



###############################################

train_x = np.array(list(train_x), dtype=float)
train_y = np.array(list(train_y), dtype=float)
valid_x = np.array(list(valid_x), dtype=float)
valid_y = np.array(list(valid_y), dtype=float)

print(train_x.shape, valid_x.shape, train_y.shape, valid_y.shape)

# encoder = MultiColumnLabelEncoder(columns=genre_strings)
# e_train_y = encoder.fit_transform(train_y)
# e_valid_y = encoder.fit_transform(valid_y)

# one_hot_encoder = preprocessing.OneHotEncoder(sparse=False)

# print("train-y shape before:", train_y.shape)
# train_y = e_train_y.reshape(-1, 1)
# valid_y = e_valid_y.reshape(-1, 1)

# print("train-y shape after:", train_y.shape)

# train_y = one_hot_encoder.fit_transform(train_y)
# valid_y = one_hot_encoder.fit_transform(valid_y)

###############################################


predictions = train_model_pipeline(ensemble.RandomForestClassifier(n_estimators=100), train_x, train_y, valid_x)
print("RF: ", metrics.accuracy_score(predictions, valid_y))
rfresults, _, _ = get_score(predictions)
print("RF score:", rfresults)


mlp_class = neural_network.MLPClassifier(hidden_layer_sizes=(200, ),
                                            solver='adam',
                                            batch_size='auto',
                                            learning_rate='adaptive', 
                                            learning_rate_init=0.0004,
                                            max_iter=900,
                                            n_iter_no_change=25)

predictions2 = train_model_pipeline(mlp_class, train_x, train_y, valid_x)
print("MLP: ", metrics.accuracy_score(predictions2, valid_y))
mlpresults, _, _ = get_score(predictions2)
print("MLP score:", mlpresults)






def Build_Model_DNN_Text(shape, nClasses, dropout=0):
    """
    buildModel_DNN_Tex(shape, nClasses,dropout)
    Build Deep neural networks Model for text classification
    Shape is input feature space
    nClasses is number of classes
    """

    model = Sequential()

    node = 16 # number of nodes
    nLayers = 1 # number of  hidden layers

    model.add(Dense(node,input_dim=shape,activation='relu'))
    model.add(Dropout(dropout))
    for i in range(0,nLayers):
        model.add(Dense(node,input_dim=node,activation='relu'))
        model.add(Dropout(dropout))
    model.add(Dense(nClasses, activation='sigmoid'))

    opt = adam_v2.Adam(learning_rate=1e-3)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


# X_train_tfidf,X_test_tfidf = TFIDF(train_x, valid_x, ngram_range=(1,3))

model_DNN = Build_Model_DNN_Text(train_x.shape[1], nClasses=22, dropout=0)
model_DNN.fit(train_x, train_y,
                              validation_data=(valid_x, valid_y),
                              epochs=5,
                              batch_size=32,
                              verbose=2)

predict_x = model_DNN.predict(valid_x)


nnresults, _, _ = get_score(predict_x)
print("DNN score:", nnresults)



classes_x=np.argmax(predict_x,axis=1)

print(metrics.classification_report(valid_y, classes_x))
print("DNN: ", metrics.accuracy_score(valid_y, classes_x))
print()