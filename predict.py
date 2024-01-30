import pickle

cv = pickle.load(open('dataset/cv_gabungan.pkl', 'rb'))
clf = pickle.load(open('dataset/clf_gabungan.pkl', 'rb'))

def predict_email(input_text):
    input_vector = cv.transform([input_text])
    prediction = clf.predict(input_vector)
    probs = clf.predict_proba(input_vector)

    return prediction[0], probs[0]

input_email = """

"""

result, confidence = predict_email(input_email)

if result == 1:
    print('This Email is spam')
else:
    print('This email is not spam')

print('Confidence Score: ')
print('Spam Score: ', confidence[1])
print('Not Spam Score: ', confidence[0])