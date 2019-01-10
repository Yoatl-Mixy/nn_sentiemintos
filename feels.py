train_big=['busco casa en polanco',
'eres muy amable',
'que es paython',
'dame un sandwich',
'eres boba',
'eres skynet']

target_big=[0,1,2,3,4,5]

clase=[('busco casa en polanco','servicial'),
('eres muy amable','alago'),
('que es paython','buscando'),
('dame un sandwich','duda'),
('eres boba','enojada'),
('eres skynet','poker_face')]

for tupla in clase :
	if tupla[1]=='servicial' :  
		target_big.append(0)
		train_big.append(tupla[0])
	elif tupla[1]=='alago' :
		target_big.append(1)
		train_big.append(tupla[0])
	elif tupla[1]=='buscando' :
		target_big.append(2)
		train_big.append(tupla[0])
	elif tupla[1]=='duda':
		target_big.append(3)
		train_big.append(tupla[0])	
	elif tupla[1]=='enojada':
		target_big.append(4)
		train_big.append(tupla[0])
	elif tupla[1]=='poker_face':
		target_big.append(5)
		train_big.append(tupla[0])

print(len(train_big))
print(len(target_big))
print('clase_1: ',len(clase))

#Entrenamiento
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib


text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                            alpha=1e-3, random_state=42,
                                            max_iter=5, tol=None)),
 ])
text_clf.fit(train_big, target_big) 
vec=CountVectorizer()
joblib.dump(text_clf, "Clasificacion_link_home_revimex.pkl")

while 1:
	x = input('Usuario: ')
	otherAnswer = text_clf.predict([x])
	if (otherAnswer == 0):
		asw="servicial (U u U)"
	elif (otherAnswer == 1):
		asw="alago (O//w//O)"
	elif (otherAnswer == 2):
		asw="buscando (o w O)?"
	elif (otherAnswer == 3):
		asw="duda (o 3 o)?"
	elif (otherAnswer == 4):
		asw="enojada (> n <)"
	elif (otherAnswer == 5):
		asw="poker_face (O __o)?"

	print('accion_',otherAnswer,'Expresion:', asw)