#Implementing regularized Non-negative Matrix factorization using Regularized gradient descent
__author__ = 'vardhaman'
import sys, numpy as np
from numpy import genfromtxt
import codecs
from numpy import linalg as LA
import numpy

#build movie dicitionary with line no as numpy movie id ,its actual movie id as the key.
def build_movies_dict(movies_file):
    i = 0
    movie_id_dict = {}
    with codecs.open(movies_file, 'r', 'latin-1') as f:
        for line in f:
            if i == 0:
                i = i+1
            else:
                movieId,title,genres = line.split(',')
                movie_id_dict[int(movieId)] = i-1
                i = i +1
    return movie_id_dict

#Each line of i/p file represents one tag applied to one movie by one user,
#and has the following format: userId,movieId,tag,timestamp
#make sure you know the number of users and items for your dataset
#return the sparse matrix as a numpy array
def read_data(input_file,movies_dict):
    #no of users
    users = 718
    #users = 5
    #no of movies
    movies = 8927
    #movies = 135887
    X = np.zeros(shape=(users,movies))
    i = 0
    #X = genfromtxt(input_file, delimiter=",",dtype=str)
    with open(input_file,'r') as f:
        for line in f:
            if i == 0:
                i = i +1
            else:
                #print "i is",i
                user,movie_id,rating,timestamp = line.split(',')
                #get the movie id for the numpy array consrtruction
                id = movies_dict[int(movie_id)]
                #print "user movie rating",user, movie, rating, i
                X[int(user),id] = float(rating)
                i = i+1
    return X

# non negative regulaized matrix factorization implemention
def matrix_factorization_not_perfect(X,P,Q,K,steps,alpha,beta):
    Q = Q.T
    for step in xrange(steps):
        print step
        #for each user
        for i in xrange(X.shape[0]):
            #for each item
            for j in xrange(X.shape[1]):
                if X[i][j] > 0 :

                    #calculate the error of the element
                    eij = X[i][j] - np.dot(P[i,:],Q[:,j])
                    #second norm of P and Q for regularilization
                    sum_of_norms = 0
                    #for k in xrange(K):
                    #    sum_of_norms += LA.norm(P[:,k]) + LA.norm(Q[k,:])
                    #added regularized term to the error
                    sum_of_norms += LA.norm(P) + LA.norm(Q)
                    #print sum_of_norms
                    eij += ((beta/2) * sum_of_norms)
                    #print eij
                    #compute the gradient from the error
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * ( 2 * eij * Q[k][j] - (beta * P[i][k]))
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - (beta * Q[k][j]))

        #compute total error
        error = 0
        #for each user
        for i in xrange(X.shape[0]):
            #for each item
            for j in xrange(X.shape[1]):
                if X[i][j] > 0:
                    error += np.power(X[i][j] - np.dot(P[i,:],Q[:,j]),2)
        if error < 0.001:
            break
    return P, Q.T
    
def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    Q = Q.T
    print 'Steps: ',
    for step in xrange(steps):
        print '{} '.format(step),
        
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    eij = R[i][j] - numpy.dot(P[i,:],Q[:,j])
                    for k in xrange(K):
                        P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
        eR = numpy.dot(P,Q)
        e = 0
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] > 0:
                    e = e + pow(R[i][j] - numpy.dot(P[i,:],Q[:,j]), 2)
                    for k in xrange(K):
                        e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
        if e < 0.001:
            break
    #print 'R='
    #print R
    #print 'P='
    #print P
    #print 'Q='
    #print Q.T    
    print ' '
    return P, Q.T

# Estimate the best values for Steps, alpba and beta
def estimate(X,K):
    values = ((10, 0.0002, 0.02),(20, 0.0002, 0.02),(40, 0.0002, 0.02),(10, 0.0004, 0.02),(10, 0.0001, 0.02))
    rmse = []
    for i in xrange(len(values)):
        N= X.shape[0]
        #no of movies
        M = X.shape[1]
        #P: an initial matrix of dimension N x K, where is n is no of users and k is hidden latent features
        P = np.random.rand(N,K)
        #Q : an initial matrix of dimension M x K, where M is no of movies and K is hidden latent features
        Q = np.random.rand(M,K)
        #steps : the maximum number of steps to perform the optimisation, hardcoding the values
        #alpha : the learning rate, hardcoding the values
        #beta  : the regularization parameter, hardcoding the values
        steps = values[i][0]
        alpha = values[i][1]
        beta = float(values[i][2])
        print 'Estimating MF with steps={}, alpha={}, beta={}, Number of users={}, Number of Movies={}\n'.format(steps, alpha, beta, N, M)
        estimated_P, estimated_Q = matrix_factorization(X,P,Q,K,steps,alpha,beta)
        #Predicted numpy array of users and movie ratings
        modeled_X = np.dot(estimated_P,estimated_Q.T)
        
        # Find RMSE
        R_actual = np.ma.masked_equal(X, 0)
        missing_mask = np.ma.getmaskarray(R_actual)
        nR_actual = np.ma.masked_array(modeled_X, mask=missing_mask)
        fit_error = nR_actual - R_actual 
        fit_error_filled = fit_error.filled(-999)
        actual_ratings = np.where(fit_error_filled > -999)
        fit_diffs = np.asarray(fit_error[actual_ratings])
        fit_RMSE = np.sqrt(np.sum(fit_diffs**2) / fit_diffs.size)
        print 'Factorization RMSE: %.3f' % fit_RMSE
        rmse.append(fit_RMSE)
        model_file = 'mf_model_mrse_{}.txt'.format('%.3f' % fit_RMSE)
        print '\nSaving Model file {}...'.format(model_file)
        np.savetxt(model_file, modeled_X, delimiter=',')
    # find best RMSE
    best_RMSE = min(rmse)
    model_file = 'mf_model_mrse_{}.txt'.format('%.3f' % best_RMSE)
    print 'Best RMSE found to be {}'.format(best_RMSE)
    print 'Use the file {} for generating suggestions...'.format(model_file)
    
def main(X,K):
    #no of users
    N= X.shape[0]
    #no of movies
    M = X.shape[1]
    #P: an initial matrix of dimension N x K, where is n is no of users and k is hidden latent features
    P = np.random.rand(N,K)
    #Q : an initial matrix of dimension M x K, where M is no of movies and K is hidden latent features
    Q = np.random.rand(M,K)
    #steps : the maximum number of steps to perform the optimisation, hardcoding the values
    #alpha : the learning rate, hardcoding the values
    #beta  : the regularization parameter, hardcoding the values
    steps = 20
    alpha = 0.0002
    beta = float(0.02)
    estimated_P, estimated_Q = matrix_factorization(X,P,Q,K,steps,alpha,beta)
    #Predicted numpy array of users and movie ratings
    modeled_X = np.dot(estimated_P,estimated_Q.T)
    
    # Find RMSE
    print 'Calculating RMSE...\n'
    R_actual = np.ma.masked_equal(X, 0)
    missing_mask = np.ma.getmaskarray(R_actual)
    nR_actual = np.ma.masked_array(modeled_X, mask=missing_mask)
    fit_error = nR_actual - R_actual 
    fit_error_filled = fit_error.filled(-999)
    actual_ratings = np.where(fit_error_filled > -999)
    fit_diffs = np.asarray(fit_error[actual_ratings])
    fit_RMSE = np.sqrt(np.sum(fit_diffs**2) / fit_diffs.size)
    
    print 'Factorization RMSE: %.3f' % fit_RMSE
    
    np.savetxt('mf_result.txt', modeled_X, delimiter=',')

if __name__ == '__main__':
    #MatrixFactorization.py <rating file>  <no of hidden features>  <movie mapping file>
    if len(sys.argv) == 4:
        ratings_file =  sys.argv[1]
        no_of_features = int(sys.argv[2])
        movies_mapping_file = sys.argv[3]

        #build a dictionary of movie id mapping with counter of no of movies
        movies_dict = build_movies_dict(movies_mapping_file)
        #read data and return a numpy array
        numpy_arr = read_data(ratings_file,movies_dict)
        #main function
        #main(numpy_arr,no_of_features)
        estimate(numpy_arr,no_of_features)
