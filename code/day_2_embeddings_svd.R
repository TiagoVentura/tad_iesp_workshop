##############################################################################
# File-Name: day2_word_embeddings.r
# Date: July 8, 2025
# author: Tiago Ventura
# topics: unsupervised learning
# Machine: MacOS High Sierra
##############################################################################

# 0 - Introduction ----------------------------------------------------------

## In this code, we will learn how to work with word embeddings in R. 
## This tutorial is inspired by these materials: 

# Chris Bail: https://cbail.github.io/textasdata/word2vec/rmarkdown/word2vec.html
# Emil Hvitfeldt and Julia Silge: https://smltar.com/
# Chris Barrie: https://cjbarrie.github.io/CTA-ED/exercise-6-unsupervised-learning-word-embedding.html
# Pablo Barbera: http://pablobarbera.com/POIR613/code.html


## This code will focus on: 

# -  Generate word embeddings using a simple combination of co-occurence matrix and matrix factorization
# -  Train a local word embedding model via Glove (also co-occurence matrix) and Word2Vec Algorithms (Neural Nets)
# -  Load pre-trained embeddings
# -  Visualize and inspect results
# -  Use embeddings in downstream supervised learning tasks


# Setup

library(tidyverse) # loads dplyr, ggplot2, and others
library(stringr) # to handle text elements
library(tidytext) # includes set of functions useful for manipulating text
library(text2vec) # for word embedding implementation
library(widyr) # for reshaping the text data
library(irlba) # for svd
library(here)
library(quanteda)

# Data
## As in Emil Hvitfeldt and Julia Silge, we will use data from 
## United States Consumer Financial Protection Bureau (CFPB) about Complains on financial products and services. 

## You can download the data here: https://github.com/EmilHvitfeldt/smltar/blob/master/data/complaints.csv.gz

cpts <- read_csv(here("code","day_2_data", "complaints.csv"))

# create an id
cpts$id <- 1:nrow(cpts)

# 1 - Generate Embeddings via Matrix Factorization -----------------------------------------

## this is not the estimation technique we saw in class. But it is a efficient way, similar to the Glove algorithm, for you to
## estimate embeddings locally. More importanly, by estimating your embeddings step by step, I think you will get a better sense 
## of how word vectors work. 

## Here are the steps: 

## Step 1: get the unigram probability for each word: How often do I see word1 and word2 independently?

## Step 2: Skipgram Probability. How often did I see word1 nearby word2? 

## Step 3: Calculate the Normalized Skipgram Probability (or PMI, as we saw before). log (p(word1, word2) / p(word1) / p(word2))

## Step 4: Convert this to a huge matrix with PMI in the cells

## Step 5: Use Singular Value Decomposition to find the word vectors (the rows on the left singular vectors matrix)


## 1.1 - Estimation ----------------
cpts <- cpts %>% slice(1:10000)

# Step 1: get the unigram probabilities

#calculate unigram probabilities (used to normalize skipgram probabilities later)

unigram_probs <- cpts %>%
    select(id,consumer_complaint_narrative) %>% 
    unnest_tokens(word, consumer_complaint_narrative) %>%
    count(word, sort = TRUE) %>%
    # calculate probabilities
    mutate(p = n / sum(n))

## Step 2: Skipgram Probabilities

#create context window with length 6
tidy_skipgrams <- cpts %>% 
    select(id, consumer_complaint_narrative) %>%
    # unnesting the ngrams
    unnest_tokens(ngram, consumer_complaint_narrative, token = "ngrams", n = 6) %>%
    # creating an id for ngram
    mutate(ngramID = row_number()) %>% 
    # create a new id which is pasting id for the comment and id for the ngram
    tidyr::unite(skipgramID, id, ngramID) %>%
    # unnesting again
    unnest_tokens(word, ngram)

# let's see how it looks like
head(tidy_skipgrams, n=20)

## What we need to do now is to calculate the joint probability of word 1 and word 2 across all the windows. 
## basically for every window

tidy_skipgrams <- tidy_skipgrams %>%
    pairwise_count(word, skipgramID, diag = TRUE, sort = TRUE) %>% # diag = T means that we also count when the word appears twice within the window
    mutate(p = n / sum(n))

## Step 3: Get the PMI
head(tidy_skipgrams)
## Join the skipgram with the unigram probabilities
normalized_prob <- tidy_skipgrams %>%
    filter(n > 20) %>%
    rename(word1 = item1, word2 = item2) %>%
    left_join(unigram_probs %>%
                  select(word1 = word, p1 = p),
              by = "word1") %>%
    left_join(unigram_probs %>%
                  select(word2 = word, p2 = p),
              by = "word2") %>%
    mutate(p_together = p / p1 / p2)

## log the final probability
pmi_matrix <- normalized_prob %>%
    mutate(pmi = log10(p_together)) 


## Step 4 - Convert to a huge matrix
pmi_matrix <- pmi_matrix %>%
    cast_sparse(word1, word2, pmi)

#remove missing data
# notice this is a non-standard list in R. It is called S4 object type, and you access the elements using @, instead of $
sum(is.na(pmi_matrix@x))
pmi_matrix@x[is.na(pmi_matrix@x)] <- 0

## Step 5 - Matrix Factorization

# run SVD
pmi_svd <- irlba(pmi_matrix, 256, maxit = 500)

str(pmi_svd)
# Here are your word vectors
word_vectors <- pmi_svd$u
rownames(word_vectors) <- rownames(pmi_matrix)

# let's look at them briefly
word_vectors["error",]


## 1.2 - Analyzing Word Embeddings ----------------

# Let's write a function to get the neared neighbors
nearest_words <- function(word_vectors, word, n){
    selected_vector = word_vectors[word,]
    
    mult = as.data.frame(word_vectors %*% selected_vector) # dot product in R
    mult %>%
        rownames_to_column() %>%
        rename(word = rowname,
               similarity = V1) %>%
        anti_join(get_stopwords(language = "en")) %>%
        arrange(-similarity) %>%
        slice(1: n)
    
}

rownames(word_vectors)
# See some words
nearest_words(word_vectors, "error", 10) 
nearest_words(word_vectors, "month", 10) 
nearest_words(word_vectors, "fee", 10) 

# Visualize these vectors
nearest_words(word_vectors, "error", 15) %>%
    mutate(token = reorder(word, similarity)) %>%
    ggplot(aes(token, similarity)) +
    geom_col(show.legend = FALSE, fill="#336B87")  +
    coord_flip() +
    theme_minimal()

# Since we have found word embeddings via singular value decomposition,
# we can use these vectors to understand what principal components explain the most variation
#in the CFPB complaints. 

# convert to a dataframe
wv_tidy <- word_vectors %>%
    as_tibble() %>%
    mutate(word=rownames(word_vectors)) %>%
    pivot_longer(cols = contains("V"), 
                 names_to = "dimension", 
                 values_to = "value") %>%
    mutate(dimension=str_remove(dimension, "V"), 
           dimension=as.numeric(dimension))


wv_tidy %>%
    # 12 largest dimensions
    filter(dimension <= 12) %>%
    # remove stop and functional words
    anti_join(get_stopwords(), by = "word") %>%
    filter(word!="xxxx", word!="xx") %>%
    # group by dimension
    group_by(dimension) %>%
    top_n(12, abs(value)) %>%
    ungroup()  %>%
    mutate(item1 = reorder_within(word, value, dimension)) %>%
    ggplot(aes(item1, value, fill = dimension)) +
    geom_col(alpha = 0.8, show.legend = FALSE) +
    facet_wrap(~dimension, scales = "free_y", ncol = 4) +
    scale_x_reordered() +
    coord_flip() +
    labs(
        x = NULL,
        y = "Value",
        title = "First 24 principal components for text of CFPB complaints",
        subtitle = paste("Top words contributing to the components that explain",
                         "the most variation")
    )



## visualize two main dimensions for a certain groups of words

#grab 100 words
forplot<-as.data.frame(word_vectors[200:300,])
forplot$word<-rownames(forplot)

#now plot
library(ggplot2)
ggplot(forplot, aes(x=V1, y=V2, label=word))+
    geom_text(aes(label=word),hjust=0, vjust=0, color="#336B87")+
    theme_minimal()+
    xlab("First Dimension Created by SVD")+
    ylab("Second Dimension Created by SVD")
