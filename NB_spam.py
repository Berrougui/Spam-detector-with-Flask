from flask import Flask, render_template, url_for, request
import pickle
import pandas as pd
import numpy as py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

app= Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route('/predict', methods= ['POST'] )
def predict():

        df= pd.read_csv("spam.csv", encoding= "latin-1") # encoding= UTF8
        #print(df.head())

        #supprimer les colonnes 
        df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
        #print(df.head())

        # ajouter une colonne label 
        df["label"]= df["v1"].map({"ham": 0, "spam": 1})

        #supprimer la colonne v1
        df.drop(['v1'], axis=1, inplace= True)

        # renomer la colonne v2 par message 
        df2=df.rename(columns= {"v2":"message"})

        X= df2["message"]
        y= df2["label"]

        #vectoriser les messages 
        cv= CountVectorizer()

        #fit transform
        X= cv.fit_transform(X)

        X_train, X_test, y_train, y_test= train_test_split(X,y, test_size= 0.33, random_state= 42)

        #Naive bayes classifier
        clf= MultinomialNB()
        clf.fit(X_train, y_train)
        clf.score(X_test, y_test)
        #print(score)

        y_pred= clf.predict(X_test)
        #print(classification_report(y_test,y_pred))

        #enregistrer le modele realise
        #joblib.dump(clf,"NB_spam_model.pkl")
        if request.method == 'POST':
            message = request.form['message']
            data = [message]
            vect = cv.transform(data).toarray()
            my_prediction = clf.predict(vect)
	    #return render_template('result.html', prediction = my_prediction)
        return render_template('result.html',prediction = my_prediction)


if __name__ == "__main__":
    app.run(debug=True)