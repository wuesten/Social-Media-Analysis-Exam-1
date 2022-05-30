from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
import re
import nltk


class Helper:
    def cross_validate_setting(X, y, model, params):
        """Form a cross validate setting.

        Keywords arguments:
        X       -- input data as dataframe
        y       -- target as dataframe
        model   -- classifier as estimator object
        params  -- hyperparameter as dict
        """
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=1)
        clf = GridSearchCV(model, params, scoring="accuracy", cv=cv, n_jobs=-1)
        search = clf.fit(X, y)
        return search

    def normalize_document(doc):
        stop_words = nltk.corpus.stopwords.words("english")
        wpt = nltk.WordPunctTokenizer()
        doc = re.sub(r"[^\w\s]", "", doc)
        doc = doc.lower()
        doc = doc.strip()
        tokens = wpt.tokenize(doc)
        filtered_tokens = [token for token in tokens if token not in stop_words]
        doc = " ".join(filtered_tokens)
        return doc
