# IMPORTS
import nltk
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import ssl

# DISABLING SSL CHECK -> (https://stackoverflow.com/questions/38916452/nltk-download-ssl-certificate-verify-failed)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# DOWNLOAD PUNKT
nltk.download('punkt')

# IMPORT SNOWBALLSTEMMER (for example, in above code DESTABILIZED is stemmed to DEST)
# SE USA PARA CONVERTIR UNA PALABRA EN SU RAIZ MÍNIMA
from nltk.stem import SnowballStemmer
stemmer = SnowballStemmer('spanish')

# OPEN CHATBOT.TXT FILE AND CLEAN IT TO SAVE THE NEW CONVERSATION
file = open("CHATBOT.txt","r+")
file.truncate(0)
file.close()

# OPEN DIALOG.JSON AND STORE ITS CONTENT IN DATA
with open("dialog.json") as file:
    data = json.load(file)

#
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

#LOOP EACH DIALOG
    for dialog in data["dialog"]:

#LOOP EACH PATTERN        
        for pattern in dialog["patterns"]:

#WE TOKENIZE EACH PATTERN IN WRDS EJ: (['Hay', 'alguien', 'ahí', '?'])
            wrds = nltk.word_tokenize(pattern)

#WE ADD ARRAY CONTENT TO WORDS
            words.extend(wrds)

#WE ADD THE ARRAY IN DOCS_X            
            docs_x.append(wrds)

#WE ADDA THE TAG IN DOCS_Y
#THEN WE HAVE THE PATTERNS DOCS_X WITH ITS ASSOCIATE TAG DOCS_Y            
            docs_y.append(dialog["tag"])

#WE ADD THE TAGS IN LABELS
        if dialog["tag"] not in labels:
            labels.append(dialog["tag"])

# WORDS = ['Hola', '!', 'Hola', '!', 'Cómo', 'estás', '?' (...)]
# STEMMER_WORDS = ['hol', '!', 'com', 'estas', (...)]
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]

# SORT AND DELETE DUPLICATES
    words = sorted(list(set(words)))

# SORT LABELS
    labels = sorted(labels)
