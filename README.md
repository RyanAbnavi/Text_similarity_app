# Fetch Rewards Coding Exercise - Text Similarity
This is a python based app to calculate the similarity between two text.
in this assignment I didn't use any library to find similarity between texts. 

# answer to questions:
1. Do you count punctuation or only words?
	@ PUNCUATIONS ARE REMOVED IN MY PROCESS

2.Which words should matter in the similarity comparison?
	@ STOP WORDS WERE ALSO REMOVED 

3. Do you care about the ordering of words?
	@ NOT IN THIS SOLUTION. 

4. What metric do you use to assign a numerical value to the similarity?
	@ COSIN SIMILARITY [0-1],  0: NOT SIMILAR AT ALL, 1: IDENTICAL

5.What type of data structures should be used?
	@ I HAVE USED DICTIONARIES AND LIST
 
# To run the app:
1. clone the repo onto your local machine: git clone https://github.com/RyanAbnavi/Text_similarity_app
2. Set the environment: CD "to the app directory" / pip install -r /path/to/requirements.txt
3. Run the Code: python app.py

There is an option for REMOVING STOP WORDS and also another option for using TF-TDF 
VECTORIZER. if tf-idf set to 'No' then CountVectorizer will be used. 


