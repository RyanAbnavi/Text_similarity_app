# import necessary libraries
from flask import Flask, render_template, request, jsonify
from similarity import Tokenizer, TextSimilarity

# create instance of Flask app
app = Flask(__name__)


# create route that renders index.html template
@app.route("/")
def home():
    return render_template("index.html")


# Bonus add a new route
@app.route("/results", methods=['POST'])
def get_similarity():
    req = [v for v in request.form.values()]
    text1 = req[0]
    text2 = req[1]
    RemoveStopWords = bool(int(req[2]))
    use_tfidf = bool(int(req[3]))
    docs = [text1, text2]
    sim = TextSimilarity(docs, RemoveStopWords=RemoveStopWords,
                         use_tfidf=use_tfidf).get_similarities()
    sim = round(sim['Text0-Text1'], 2)
    tokenizer = Tokenizer(text1)
    text1_total_words, text1_stop_words = tokenizer.total_words, tokenizer.stopWords

    tokenizer = Tokenizer(text2)
    text2_total_words, text2_stop_words = tokenizer.total_words, tokenizer.stopWords

    return render_template("results.html",  text1_total_words=text1_total_words, text1_stop_words=text1_stop_words,
                           text2_total_words=text2_total_words, text2_stop_words=text2_stop_words, sim=sim)


if __name__ == "__main__":
    app.run(debug=True)
