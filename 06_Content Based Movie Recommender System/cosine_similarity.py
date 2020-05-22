# author: Shivam

# Using sk-learn to count the frequency of word present in the sentence
from sklearn.feature_extraction.text import CountVectorizer


# Using sk-learn to find similarity between these 2 texts.
from sklearn.metrics.pairwise import cosine_similarity


text = ["Shivam India India","India Shivam Shivam"]

# [[2-India,1-Shivam],[1-Shivam,2-India]]

cv = CountVectorizer()

countMatrix = cv.fit_transform(text)

print(countMatrix.toarray())
# [[2 1] [1 2]]

# finding similarity
similarity_score = cosine_similarity(countMatrix)

print(similarity_score)
# [[1.  0.8]
# [0.8 1. ]]

# 1.0 shows that 1st sentence is 100% similar to the 1st sentence.
# 0.8 shows that 1 st sentence is 80% similar to 2nd sentence.
