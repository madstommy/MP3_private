import numpy as np
import math


def normalize(input_matrix):
    """
    Normalizes the rows of a 2d input_matrix so they sum to 1
    """

    row_sums = input_matrix.sum(axis=1)
    try:
        assert (np.count_nonzero(row_sums)==np.shape(row_sums)[0]) # no row should sum to zero
    except Exception:
        raise Exception("Error while normalizing. Row(s) sum to zero")
    new_matrix = input_matrix / row_sums[:, np.newaxis]
    return new_matrix

       
class Corpus(object):

    """
    A collection of documents.
    """

    def __init__(self, documents_path):
        """
        Initialize empty document list.
        """
        self.documents = []
        self.vocabulary = []
        self.likelihoods = []
        self.documents_path = documents_path
        self.term_doc_matrix = None 
        self.document_topic_prob = None  # P(z | d)
        self.topic_word_prob = None  # P(w | z)
        self.topic_prob = None  # P(z | d, w)

        self.number_of_documents = 0
        self.vocabulary_size = 0

    def build_corpus(self):
        """
        Read document, fill in self.documents, a list of list of word
        self.documents = [["the", "day", "is", "nice", "the", ...], [], []...]
        
        Update self.number_of_documents
        """
        # #############################
        # your code here
        # #############################
        file = open(self.documents_path, 'r')
        for line in file:
            new_line = line.split()
            if new_line[0] == '0' or new_line[0] == '1':
                self.documents.append(new_line[1:-1])
            else:
                self.documents.append(new_line)
        self.number_of_documents = len(self.documents)
        
    def build_vocabulary(self):
        """
        Construct a list of unique words in the whole corpus. Put it in self.vocabulary
        for example: ["rain", "the", ...]

        Update self.vocabulary_size
        """
        # #############################
        # your code here
        # #############################
        vocab_set = set()
        for doc in self.documents:
            for word in doc:
                vocab_set.add(word)
        
        self.vocabulary = list(vocab_set)
        self.vocabulary_size = len(self.vocabulary)

    def build_term_doc_matrix(self):
        """
        Construct the term-document matrix where each row represents a document, 
        and each column represents a vocabulary term.

        self.term_doc_matrix[i][j] is the count of term j in document i
        """
        # ############################
        # your code here
        # ############################
        self.term_doc_matrix = np.zeros( (len(self.documents), self.vocabulary_size))

        for i in range(len(self.documents)):
            for word in self.documents[i]:
                j = self.vocabulary.index(word)
                self.term_doc_matrix[i][j]+=1

    def initialize_randomly(self, number_of_topics):
        """
        Randomly initialize the matrices: document_topic_prob and topic_word_prob
        which hold the probability distributions for P(z | d) and P(w | z): self.document_topic_prob, and self.topic_word_prob

        Don't forget to normalize! 
        HINT: you will find numpy's random matrix useful [https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.random.html]
        """
        # ############################
        # your code here
        # ############################
        self.document_topic_prob = np.random.random( (len(self.documents), number_of_topics) )
        self.topic_word_prob = np.random.random( (number_of_topics, len(self.vocabulary) ))

        self.document_topic_prob = normalize(self.document_topic_prob)
        self.topic_word_prob = normalize(self.topic_word_prob)
        

    def initialize_uniformly(self, number_of_topics):
        """
        Initializes the matrices: self.document_topic_prob and self.topic_word_prob with a uniform 
        probability distribution. This is used for testing purposes.

        DO NOT CHANGE THIS FUNCTION
        """
        self.document_topic_prob = np.ones((self.number_of_documents, number_of_topics))
        self.document_topic_prob = normalize(self.document_topic_prob)

        self.topic_word_prob = np.ones((number_of_topics, len(self.vocabulary)))
        self.topic_word_prob = normalize(self.topic_word_prob)

    def initialize(self, number_of_topics, random=False):
        """ Call the functions to initialize the matrices document_topic_prob and topic_word_prob
        """
        print("Initializing...")

        if random:
            self.initialize_randomly(number_of_topics)
        else:
            self.initialize_uniformly(number_of_topics)

    def expectation_step(self):
        """ The E-step updates P(z | w, d)
        """
        #print("E step:")
        
        # ############################
        # your code here
        # ############################
        for i in range(len(self.document_topic_prob)):
            scalar_multiples = []
            for j in range(len(self.document_topic_prob[0])):
                pi_d_j = self.document_topic_prob[i][j]
                one_row = pi_d_j * self.topic_word_prob[j]
                scalar_multiples.append(one_row)
            self.topic_prob[i] =  np.transpose(normalize(np.transpose(scalar_multiples)))
            

    def maximization_step(self, number_of_topics):
        """ The M-step updates P(w | z)
        """
        #print("M step:")
        
        # update P(w | z)
        
        # ############################
        # your code here
        # ############################
        for i in range(len(self.term_doc_matrix[0])):
            word_counts = self.term_doc_matrix[ :,i]
            for j in range(number_of_topics):
                word_probs = self.topic_prob[ :, j, i]
                self.topic_word_prob[j][i] = np.dot(word_counts, word_probs)
        self.topic_word_prob = normalize(self.topic_word_prob)
        
        # update P(z | d)

        # ############################
        # your code here
        # ############################
        for i in range(len(self.document_topic_prob)):
            word_counts = self.term_doc_matrix[i]
            for j in range(number_of_topics):
                word_probs = self.topic_prob[i][j]
                self.document_topic_prob[i][j] = np.dot(word_counts, word_probs)
        self.document_topic_prob = normalize(self.document_topic_prob)



    def calculate_likelihood(self, number_of_topics):
        """ Calculate the current log-likelihood of the model using
        the model's updated probability matrices
        
        Append the calculated log-likelihood to self.likelihoods

        """
        # ############################
        # your code here
        # ############################
        result = 0
        for i in range(len(self.document_topic_prob)):
            word_counts = self.term_doc_matrix[i]
            for j in range(len(word_counts)):
                pi_d_j = self.topic_prob[i, :, j]
                word_probs = self.topic_word_prob[:, j]
                result += word_counts[j] * math.log(np.dot(pi_d_j, word_probs), 2)
        
        self.likelihoods.append(result)
        return result

    def plsa(self, number_of_topics, max_iter, epsilon):

        """
        Model topics.
        """
        print ("EM iteration begins...")
        
        # build term-doc matrix
        self.build_term_doc_matrix()
        
        # Create the counter arrays.
        
        # P(z | d, w)
        self.topic_prob = np.zeros([self.number_of_documents, number_of_topics, self.vocabulary_size], dtype=np.float)

        # P(z | d) P(w | z)
        self.initialize(number_of_topics, random=True)

        # Run the EM algorithm
        current_likelihood = 0.0
        previous_likelihood = 0.0

        for iteration in range(max_iter):
            print("Iteration #" + str(iteration + 1) + "...")

            # ############################
            # your code here
            # ############################
            self.expectation_step()
            self.maximization_step(number_of_topics)
            previous_likelihood = current_likelihood
            current_likelihood = self.calculate_likelihood(number_of_topics)
            print(current_likelihood)
            if abs(current_likelihood - previous_likelihood) < epsilon:
                break
        print("Algorithm Complete. Final Likelihood is {}".format(current_likelihood))


def main():
    documents_path = 'data/test.txt'
    corpus = Corpus(documents_path)  # instantiate corpus
    corpus.build_corpus()
    corpus.build_vocabulary()
    print(corpus.vocabulary)
    print(corpus.term_doc_matrix)
    print("Vocabulary size:" + str(len(corpus.vocabulary)))
    print("Number of documents:" + str(len(corpus.documents)))
    number_of_topics = 2
    max_iterations = 50
    epsilon = 0.001
    corpus.plsa(number_of_topics, max_iterations, epsilon)



if __name__ == '__main__':
    main()
