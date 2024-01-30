import pickle

cv = pickle.load(open('dataset/cv_gabungan.pkl', 'rb'))
clf = pickle.load(open('dataset/clf_gabungan.pkl', 'rb'))

def predict_email(input_text):
    input_vector = cv.transform([input_text])
    prediction = clf.predict(input_vector)
    return prediction[0]

input_email = """
Subject: re : research and development charges to gpg  here it is !  - - - - - - - - - - - - - - - - - - - - - - forwarded by shirley crenshaw / hou / ect on 08 / 14 / 2000  07 : 47 am - - - - - - - - - - - - - - - - - - - - - - - - - - -  vince j kaminski  08 / 10 / 2000 02 : 25 pm  to : vera apodaca / et & s / enron @ enron  cc : vince j kaminski / hou / ect @ ect , shirley crenshaw / hou / ect @ ect , pinnamaneni  krishnarao / hou / ect @ ect  subject : re : research and development charges to gpg  vera ,  we shall talk to the accounting group about the correction .  vince  08 / 09 / 2000 03 : 26 pm  vera apodaca @ enron  vera apodaca @ enron  vera apodaca @ enron  08 / 09 / 2000 03 : 26 pm  08 / 09 / 2000 03 : 26 pm  to : pinnamaneni krishnarao / hou / ect @ ect  cc : vince j kaminski / hou / ect @ ect  subject : research and development charges to gpg  per mail dated june 15 from kim watson , there was supposed to have occurred  a true - up of $ 274 . 7 in july for the fist six months of 2000 . reviewing july  actuals , i was not able to locate this entry . would you pls let me know  whether this entry was made , if not , when do you intend to process it .  thanks .

"""

result = predict_email(input_email)

if result == 1:
    print('This Email is spam')
else:
    print('This email is not spam')