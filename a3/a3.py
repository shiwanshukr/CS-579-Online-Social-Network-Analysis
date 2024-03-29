# coding: utf-8

# # Assignment 3:  Recommendation systems
#
# Here we'll implement a content-based recommendation algorithm.
# It will use the list of genres for a movie as the content.
# The data come from the MovieLens project: http://grouplens.org/datasets/movielens/
# Note that I have not provided many doctests for this one. I strongly
# recommend that you write your own for each function to ensure your
# implementation is correct.

# Please only use these imports.
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/p9wmkvbqt1xr6lc/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()


def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())


def tokenize(movies):
    """
    Append a new column to the movies DataFrame with header 'tokens'.
    This will contain a list of strings, one per token, extracted
    from the 'genre' field of each movie. Use the tokenize_string method above.

    Note: you may modify the movies parameter directly; no need to make
    a new copy.
    Params:
      movies...The movies DataFrame
    Returns:
      The movies DataFrame, augmented to include a new column called 'tokens'.

    >>> movies = pd.DataFrame([[123, 'Horror|Romance'], [456, 'Sci-Fi']], columns=['movieId', 'genres'])
    >>> movies = tokenize(movies)
    >>> movies['tokens'].tolist()
    [['horror', 'romance'], ['sci-fi']]
    """
    ###TODO
    ## tokenize
    tokenize_list = list()

    for i in movies['genres']:

        tokenize_list.append(tokenize_string(i))
    # Add a new column and then add series to it. 
    movies['tokens']= np.array(tokenize_list) 
    return movies


def featurize(movies):
    """
    Append a new column to the movies DataFrame with header 'features'.
    Each row will contain a csr_matrix of shape (1, num_features). Each
    entry in this matrix will contain the tf-idf value of the term, as
    defined in class:
    tfidf(i, d) := tf(i, d) / max_k tf(k, d) * log10(N/df(i))
    where:
    i is a term
    d is a document (movie)
    tf(i, d) is the frequency of term i in document d
    max_k tf(k, d) is the maximum frequency of any term in document d
    N is the number of documents (movies)
    df(i) is the number of unique documents containing term i

    Params:
      movies...The movies DataFrame
    Returns:
      A tuple containing:
      - The movies DataFrame, which has been modified to include a column named 'features'.
      - The vocab, a dict from term to int. Make sure the vocab is sorted alphabetically as in a2 (e.g., {'aardvark': 0, 'boy': 1, ...})
    """
    ###TODO


    '''
    
    '''
    pass

    ###TODO
    # Term ==> 'Genres' and document ==> 'Movies'
    # Term ==> 'Genres' and document ==> 'Movies'
    
    # code
    
    genres_frequency = defaultdict(lambda: 0) # For calculating documents i.e Movies
     # For calculating term in document ==> i.e Genres 
    genre_list = list(defaultdict(lambda: 0)) # Total list in each docs 
    
    
    for token in movies['tokens']:
        genre_set = set(token) 
        # Unique documents ==> documents
        for document in genre_set:
            genres_frequency[document] = genres_frequency[document]+1
        term_frequency = defaultdict(lambda:0)
        for term in token: # within that movie
            term_frequency[term] = term_frequency[term]+1
        genre_list.append(term_frequency)
    
    gen_freq = sorted(genres_frequency)
    all_features = gen_freq
    vocab = dict((k, all_features.index(k)) for k in all_features )
    # Number_of_doc is the number of documents (movies)
    Number_of_doc = movies.shape[0] 
    list_csr_matrix = []
    for i in range(0,Number_of_doc,1):
        document_term=genre_list[i]
        max_k = document_term[max(document_term, key = document_term.get)]
        data_movie = []
        row = []
        col =[]
        for i in document_term:
            if i in vocab:
                col.append(vocab[i])
                row.append(0)
                # math.log((N / genres_frequency[i]), 10)| np.log(N/genres_frequency[i])
                tf_idf = document_term[i] / max_k* math.log((Number_of_doc / genres_frequency[i]), 10)
                
                data_movie.append(tf_idf)
        doc_csr=csr_matrix((data_movie, (row, col)), shape = (1, len(vocab)))
        list_csr_matrix.append(doc_csr)
    movies['features'] =  list_csr_matrix
    
    return (movies, vocab)




def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]


def cosine_sim(a, b):
    """
    Compute the cosine similarity between two 1-d csr_matrices.
    Each matrix represents the tf-idf feature vector of a movie.
    Params:
      a...A csr_matrix with shape (1, number_features)
      b...A csr_matrix with shape (1, number_features)
    Returns:
      A float. The cosine similarity, defined as: dot(a, b) / ||a|| * ||b||
      where ||a|| indicates the Euclidean norm (aka L2 norm) of vector a.
    """
    ###TODO
    ## check...hopefully working..and fine..change more if possible
    pass
    num = a.dot(np.transpose(b)).toarray()[0][0]
    ''''
         linalg.norm : Compute the dot product of two or more arrays in a single 
        function call, while automatically selecting the fastest evaluation order.
    '''
    denom = (np.linalg.norm(a.toarray()) * np.linalg.norm(b.toarray()))

    return num/denom


def make_predictions(movies, ratings_train, ratings_test):
    """
    Using the ratings in ratings_train, predict the ratings for each
    row in ratings_test.

    To predict the rating of user u for movie i: Compute the weighted average
    rating for every other movie that u has rated.  Restrict this weighted
    average to movies that have a positive cosine similarity with movie
    i. The weight for movie m corresponds to the cosine similarity between m
    and i.

    If there are no other movies with positive cosine similarity to use in the
    prediction, use the mean rating of the target user in ratings_train as the
    prediction.

    Params:
      movies..........The movies DataFrame.
      ratings_train...The subset of ratings used for making predictions. These are the "historical" data.
      ratings_test....The subset of ratings that need to predicted. These are the "future" data.
    Returns:
      A numpy array containing one predicted rating for each element of ratings_test.
    """
    ###TODO
    features_m = dict()

    for en in range(0,len(movies.index),1):
        
        features_m[movies.at[en, 'movieId']] = movies.at[en, 'features']

    prediction_movies = list()
    for record in ratings_test.itertuples():
        # Insert userID and movieID in record index 1 and 2 
        userId = record[1]

        movieId = record[2]

        sum_similarity = list()
        all_ratings = list()
        weighted_rating = list()
        for movie_record in ratings_train.loc[ratings_train['userId'] == userId].itertuples():
            
            similarity_cosine = cosine_sim(features_m[movieId], features_m[movie_record[2]])

            w = movie_record[3]

            all_ratings.append(w)


            if similarity_cosine > 0:
                en = similarity_cosine * movie_record[3]
                sum_similarity.append(en)

                weighted_rating.append(similarity_cosine)

        res = len(sum_similarity)

        if res > 0:

            a = sum(sum_similarity)
            b = sum(weighted_rating)
            rating =  a / b
        else:

            rating = np.mean(all_ratings)
        prediction_movies.append(rating)

    arr = np.array(prediction_movies)

    return arr



def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()




def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
